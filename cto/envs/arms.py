"""
Bimanual Franka Panda + box in MuJoCo; same interface as cto/envs/fingers_mujoco.FingerDoubleAndBox.

    env = BimanualPandaAndBox(params, box_pos, box_orn, ee_init=None, render=False, dt=0.001)

Setup
  * Two Panda arms (MuJoCo Menagerie `franka_emika_panda/panda_nohand.xml`, hand removed) are mounted on
    the table at x = -/+ `base_distance`, facing the box (arm 1 is rotated by 180 deg about z).
  * Each arm carries a rigid tool ending in a small sphere (radius 10 mm, wider than the 6 mm shaft). The
    end-effector frame `tool_tip` is the sphere centre, i.e. the planner's point-contact location.
    3D position control only; the 4 redundant DoF are used for a posture task.
  * The default posture keeps the tool VERTICAL. This matters: with a tilted tool the shaft leans over the
    box edge and pinches it there (wrong contact point and height), and the box rolls over.
  * Pinocchio parses the very same MJCF, so the planner's kinematics and the simulator agree exactly.

Exposed like the finger env: arm0/arm1/arms (also finger0/finger1/fingers), ee0_id/ee1_id/ee_ids,
q_default (a fresh copy on every access: the IK in cto/kinematics.py modifies its q_init in place),
col_detectors, set_box_pose, get_box_pose, add_visual_frame, step, close, start/stop_video_recording.

Joint-limit violations of an IK result are reported by col_detectors[c].in_collision as "in collision",
because cto/kinematics.inverse_kinematics_3d does not enforce limits.

Friction is explicit (geom priority): arm tool/links <-> box = params.friction_coeff,
box <-> table = params.environment_friction, true elliptic friction cone.
Gravity of the arms is compensated with MuJoCo's bias forces (the arms are heavy, unlike the fingers).
"""
import time
from pathlib import Path

import numpy as np
import mujoco
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from cto.envs.collision import CollisionDetector

DEFAULT_PANDA_XML = "/home/tad/1_MUJOCO/MJ_GRASP/mjmanip/mjmanip/models/panda/panda_nohand.xml"
# Vertical tool (pointing straight down) hovering at (-/+0.15, 0, 0.14) for base_distance = 0.70; joint-limit margin 0.85 rad.
# A tilted tool is NOT an option: its shaft leans over the box edge and pinches it there instead of the sphere touching the face.
Q_DEFAULT = np.array([0.0, 0.318, 0.0, -2.224, 0.0, 2.541, 0.785])


def _xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]])


def _wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]])


def _build_pin_arm(panda_xml, base_xyz, base_yaw, tool_length):
    """Pinocchio model of one arm in WORLD coordinates with a `tool_tip` frame."""
    m = pin.buildModelFromMJCF(str(panda_xml))
    att = m.frames[m.getFrameId("attachment")]
    tip = pin.Frame("tool_tip", att.parentJoint, m.getFrameId("attachment"),
                    att.placement * pin.SE3(np.eye(3), np.array([0.0, 0.0, tool_length])),
                    pin.FrameType.OP_FRAME)
    fid = m.addFrame(tip)
    T = pin.SE3(pin.rpy.rpyToMatrix(0.0, 0.0, base_yaw), np.asarray(base_xyz, dtype=float))
    m.jointPlacements[1] = T * m.jointPlacements[1]  # shifts the whole arm into the world frame
    return RobotWrapper(m), fid


class Arm:
    def __init__(self, env, prefix, pin_robot, ee_id):
        self.env = env
        self.prefix = prefix
        self.pin_robot = pin_robot
        self.end_eff_ids = [ee_id]
        self.nv = pin_robot.model.nv
        self.nq = pin_robot.model.nq
        self.lower = pin_robot.model.lowerPositionLimit.copy()
        self.upper = pin_robot.model.upperPositionLimit.copy()

        model = env.model
        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}joint{i}") for i in range(1, 8)]
        assert all(j >= 0 for j in jids), "Panda joints not found"
        self.qadr = model.jnt_qposadr[jids]
        self.dadr = model.jnt_dofadr[jids]
        self.tau = np.zeros(7)
        self.geom_ids = []

    def get_state(self):
        d = self.env.data
        return d.qpos[self.qadr].copy(), d.qvel[self.dadr].copy()

    def update_pinocchio(self, q, dq):
        self.pin_robot.computeJointJacobians(q)
        self.pin_robot.framesForwardKinematics(q)

    def get_state_update_pinocchio(self):
        q, dq = self.get_state()
        self.update_pinocchio(q, dq)
        return q, dq

    def reset_state(self, q, dq):
        d = self.env.data
        d.qpos[self.qadr] = q
        d.qvel[self.dadr] = dq
        mujoco.mj_forward(self.env.model, d)

    def set_configuration(self, q):
        d = self.env.data
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.env.model, d)

    def violates_limits(self, q, tol=1e-6):
        q = np.asarray(q)
        return bool(np.any(q < self.lower - tol) or np.any(q > self.upper + tol))

    def send_joint_command(self, tau):
        self.tau = np.asarray(tau, dtype=float).copy()

    def compute_numerical_quantities(self, dt):
        pass


class BimanualPandaAndBox(object):
    def __init__(self, params, box_pos, box_orn, ee_init=None, render=False, dt=0.001,
                 panda_xml=DEFAULT_PANDA_XML, base_distance=0.70, tool_length=0.14, tool_radius=0.01,
                 joint_damping=0.1, joint_armature=0.02):
        self.dt = dt
        self.params = params
        mu_e, mu_c = params.environment_friction, params.friction_coeff
        self._q_default = Q_DEFAULT.copy()
        panda_xml = Path(panda_xml)
        if not panda_xml.exists():
            raise FileNotFoundError(
                f"{panda_xml} not found. Get it with:\n"
                "  git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie\n"
                "  cd mujoco_menagerie && git sparse-checkout set franka_emika_panda\n"
                "and copy franka_emika_panda/ to cto/envs/resources/ (or pass panda_xml=...).")

        ground = params.ground_height
        bases = [([-base_distance, 0.0, ground], 0.0), ([base_distance, 0.0, ground], np.pi)]
        prefixes = ["r0_", "r1_"]

        # ---------------------------------------------------------------- scene
        spec = mujoco.MjSpec()
        spec.option.timestep = dt
        spec.option.gravity = [0, 0, -9.81]
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        spec.option.impratio = 10
        spec.meshdir = str(panda_xml.parent / "assets")

        FINGER, WORLD = 2, 1  # arms collide with the box and the table, not with each other
        for (xyz, yaw), prefix in zip(bases, prefixes):
            child = mujoco.MjSpec.from_file(str(panda_xml))
            for a in list(child.actuators):  # we apply joint torques directly
                child.delete(a)
            for k in list(child.keys):
                child.delete(k)
            for j in child.joints:
                j.damping = [joint_damping, 0.0, 0.0]  # (array field in MjSpec; hinge joints use [0])
                j.armature = joint_armature
            frame = spec.worldbody.add_frame(pos=list(xyz), quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
            spec.attach(child, frame=frame, prefix=prefix)

        for prefix in prefixes:  # tool on the flange: shaft + sphere at the tip
            att = spec.body(prefix + "attachment")
            shaft = att.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0, 0, 0, 0, 0, tool_length],
                                 size=[0.006, 0, 0], rgba=[0.3, 0.3, 0.3, 1])
            tip = att.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, pos=[0, 0, tool_length],
                               size=[tool_radius, 0, 0], rgba=[0.9, 0.2, 0.2, 1])
            for g in (shaft, tip):
                g.mass = 0.01 if g is tip else 0.05

        for g in spec.geoms:  # all arm/tool collision geoms (visual geoms have contype 0)
            if g.contype != 0 or g.conaffinity != 0:
                g.contype, g.conaffinity, g.priority, g.condim = FINGER, WORLD, 1, 3
                g.friction = [mu_c, 0.005, 0.0001]

        plane = spec.worldbody.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE,
                                        pos=[0, 0, ground], size=[1.5, 1.5, 0.1], rgba=[0.8, 0.8, 0.8, 1])
        plane.contype, plane.conaffinity, plane.priority, plane.condim = WORLD, WORLD, 1, 3
        plane.friction = [mu_e, 0.005, 0.0001]

        box = spec.worldbody.add_body(name="box", pos=list(box_pos), quat=list(_xyzw_to_wxyz(box_orn)))
        box.add_freejoint(name="box_joint")
        box.explicitinertial = True
        box.mass = float(params.mass)
        box.ipos = [0, 0, 0]
        box.inertia = [float(x) for x in np.diag(np.asarray(params.inertia))]
        bg = box.add_geom(name="box_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                          size=[params.box.length / 2] * 3, rgba=[0.1, 0.1, 0.1, 0.6])
        bg.contype, bg.conaffinity, bg.priority, bg.condim = WORLD, WORLD | FINGER, 0, 3
        bg.friction = [mu_c, 0.005, 0.0001]

        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.box_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "box_geom")
        bj = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "box_joint")
        self.box_qadr, self.box_dadr = self.model.jnt_qposadr[bj], self.model.jnt_dofadr[bj]

        # ---------------------------------------------------------------- arms (pinocchio models share the MJCF)
        self.arms = []
        for (xyz, yaw), prefix in zip(bases, prefixes):
            pin_robot, ee_id = _build_pin_arm(panda_xml, xyz, yaw, tool_length)
            arm = Arm(self, prefix, pin_robot, ee_id)
            for i in range(1, 8):
                body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}link{i}")
                arm.geom_ids += [g for g in range(self.model.ngeom)
                                 if self.model.geom_bodyid[g] == body and self.model.geom_contype[g] != 0]
            att_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, prefix + "attachment")
            arm.geom_ids += [g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == att_body]
            self.arms.append(arm)
        self.arm0, self.arm1 = self.arms
        self.fingers = self.arms  # names used by cto/heuristics.py
        self.finger0, self.finger1 = self.arm0, self.arm1
        self.ee_ids = [a.end_eff_ids[0] for a in self.arms]
        self.ee0_id, self.ee1_id = self.ee_ids
        self.configs = [None, None]

        dq = np.zeros(7)
        if ee_init is None:
            for a in self.arms:
                a.reset_state(self.q_default, dq)
        else:
            from cto.kinematics import inverse_kinematics_3d
            for a, ee, ee_id in zip(self.arms, ee_init, self.ee_ids):
                q, _ = inverse_kinematics_3d(a.pin_robot, ee, ee_id, q_init=self.q_default,
                                             q_null=self.q_default, max_it=300)
                a.reset_state(np.clip(q, a.lower, a.upper), dq)

        self.set_box_pose(np.concatenate([box_pos, box_orn]))
        mujoco.mj_forward(self.model, self.data)
        self.col_detectors = [CollisionDetector(self, a) for a in self.arms]

        self.viewer, self._n_steps, self._frames, self._renderer = None, 0, None, None
        self._sync_every = max(1, int(round(1.0 / (60.0 * dt))))
        if render:
            import mujoco.viewer as mj_viewer
            self.viewer = mj_viewer.launch_passive(self.model, self.data)
            self.viewer.cam.distance, self.viewer.cam.azimuth, self.viewer.cam.elevation = 1.6, 90, -25
            self.viewer.cam.lookat[:] = [0, 0, 0.15]
            self.viewer.sync()

    # ------------------------------------------------------------------------------------------
    @property
    def q_default(self):
        return self._q_default.copy()

    def add_visual_frame(self, pos, orn, length=0.1, width=1):
        if self.viewer is None:
            return
        scn = self.viewer.user_scn
        for vec, rgba in ((np.array([length, 0, 0]), [1, 0, 0, 1]), (np.array([0, length, 0]), [0, 1, 0, 1]),
                          (np.array([0, 0, length]), [0, 0, 1, 1])):
            if scn.ngeom >= scn.maxgeom:
                return
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9),
                                np.array(rgba, dtype=np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.001 * width,
                                 np.asarray(pos, dtype=float), np.asarray(pos, dtype=float) + orn @ vec)
            scn.ngeom += 1
        self.viewer.sync()

    def step(self, sleep_factor=0):
        """One simulation step. Joint torques = commanded torque + gravity/Coriolis compensation."""
        for a in self.arms:
            self.data.qfrc_applied[a.dadr] = a.tau + self.data.qfrc_bias[a.dadr]
        if self.viewer is not None and not self.viewer.is_running():
            self.viewer = None
        if self.viewer is not None:
            with self.viewer.lock():
                mujoco.mj_step(self.model, self.data)
            self._n_steps += 1
            if self._n_steps % self._sync_every == 0:
                self.viewer.sync()
            if sleep_factor:
                time.sleep(sleep_factor * self.dt)
        else:
            mujoco.mj_step(self.model, self.data)
        if self._frames is not None and self.data.time >= len(self._frames) / 30.0:
            self._renderer.update_scene(self.data)
            self._frames.append(self._renderer.render())

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def set_box_pose(self, pose):
        a = self.box_qadr
        self.data.qpos[a:a + 3] = pose[:3]
        self.data.qpos[a + 3:a + 7] = _xyzw_to_wxyz(pose[3:])
        self.data.qvel[self.box_dadr:self.box_dadr + 6] = 0
        mujoco.mj_kinematics(self.model, self.data)

    def get_box_pose(self):
        a = self.box_qadr
        return pin.XYZQUATToSE3(np.concatenate([self.data.qpos[a:a + 3], _wxyz_to_xyzw(self.data.qpos[a + 3:a + 7])]))

    def start_video_recording(self, file_name):
        self._video_file = file_name
        self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._frames = []

    def stop_video_recording(self):
        if self._frames is None:
            return
        import imageio
        imageio.mimsave(self._video_file, self._frames, fps=30)
        self._renderer.close()
        self._frames, self._renderer = None, None
