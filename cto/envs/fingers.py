"""
MuJoCo version of cto/envs/fingers.py (FingerDoubleAndBox).

Exposes the same interface the rest of the project uses:
    env.finger0 / env.finger1 / env.fingers   (pin_robot, end_eff_ids, nv,
                                                get_state_update_pinocchio, send_joint_command, reset_state)
    env.ee0_id / env.ee1_id / env.ee_ids, env.q_default, env.configs
    env.collision_detectors[c].in_collision(q, margin)
    env.set_box_pose / get_box_pose / add_visual_frame / step / close
    env.start_video_recording / stop_video_recording

Differences from the PyBullet environment (deliberate):
  * Friction is explicit. MuJoCo normally takes the MAX of the two geoms' coefficients (Bullet
    multiplies them), so we use geom `priority` to make the contact use exactly
        finger <-> box : params.friction_coeff
        box    <-> plane: params.environment_friction
    with a true (elliptic) Coulomb cone, like pybullet's enableConeFriction=1.
  * Joint torques are applied directly with qfrc_applied, unclipped
"""
import time
import tempfile

import numpy as np
import mujoco as mj
import pinocchio as pin

from cto.kinematics import pin_ik
from robot_properties_nyu_finger.config import NYUFingerConfig, NYUFingerDoubleConfig0, NYUFingerDoubleConfig1
from cto.envs.collision import CollisionDetector


def _xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]])


def _wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]])


class Finger:
    def __init__(self, env, config: NYUFingerConfig, prefix: str):
        self.env = env
        self.config = config
        self.pin_robot = config.pin_robot
        self.end_eff_ids = config.ee_ids
        self.nv = self.pin_robot.model.nv
        self.nq = self.pin_robot.model.nq
        self.lower = self.pin_robot.model.lowerPositionLimit.copy()
        self.upper = self.pin_robot.model.upperPositionLimit.copy()

        model = env.model
        jids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, prefix + n) for n in config.joint_names]
        assert all(j >= 0 for j in jids), "finger joints not found in the MuJoCo model"
        self.qadr = model.jnt_qposadr[jids]
        self.dadr = model.jnt_dofadr[jids]
        self.tau = np.zeros(len(jids))
        self.geom_ids = []  # collision geoms of the three moving links
        self.prefix = prefix

    # --- state -------------------------------------------------------------------------------
    def get_state(self):
        d = self.env.data
        return d.qpos[self.qadr].copy(), d.qvel[self.dadr].copy()

    def update_pinocchio(self, q, dq):
        self.pin_robot.computeJointJacobians(q)
        pin.updateFramePlacements(self.pin_robot.model, self.pin_robot.data)

    def get_state_update_pinocchio(self):
        q, dq = self.get_state()
        self.update_pinocchio(q, dq)
        return q, dq

    def reset_state(self, q, dq):
        d = self.env.data
        d.qpos[self.qadr] = q
        d.qvel[self.dadr] = dq
        mj.mj_forward(self.env.model, d)

    def set_configuration(self, q):
        """Kinematics only (cheap); used by the collision detector."""
        d = self.env.data
        d.qpos[self.qadr] = q
        mj.mj_kinematics(self.env.model, d)

    def violates_limits(self, q, tol=1e-6):
        q = np.asarray(q)
        return bool(np.any(q < self.lower - tol) or np.any(q > self.upper + tol))

    def send_joint_command(self, tau):
        self.tau = np.asarray(tau, dtype=float).copy()

    def compute_numerical_quantities(self, dt):
        pass


class FingerDoubleAndBox(object):
    def __init__(self, params, box_pos, box_orn, ee_init=None, render=False, dt=0.001):
        self.dt = dt
        self.params = params
        mu_e = params.environment_friction
        mu_c = params.friction_coeff

        configs = [NYUFingerDoubleConfig0(), NYUFingerDoubleConfig1()]
        self.configs = configs

        # ---------------------------------------------------------------- scene
        spec = mj.MjSpec()
        spec.option.timestep = dt
        spec.option.gravity = [0, 0, -9.81]
        spec.option.cone = mj.mjtCone.mjCONE_ELLIPTIC  # true Coulomb cone
        spec.option.impratio = 10  # stiffer friction
        meshdir = str(configs[0].meshes_path)
        spec.meshdir = meshdir

        self._tmp = tempfile.TemporaryDirectory()
        prefixes = ["r0_", "r1_"]
        for cfg, prefix in zip(configs, prefixes):
            child = mj.MjSpec.from_file(cfg.mjcf_path)
            child.meshdir = meshdir
            child.compiler.discardvisual = True
            child.compiler.balanceinertia = True
            spec.attach(child, frame=spec.worldbody.add_frame(), prefix=prefix)

        # contact bit masks: fingers collide with box and plane, never with each other
        FINGER, WORLD = 2, 1
        for g in spec.geoms:
            if g.name == "" and g.type == mj.mjtGeom.mjGEOM_MESH:
                g.contype, g.conaffinity = FINGER, WORLD
                g.priority = 1
                g.friction = [mu_c, 0.005, 0.0001]
                g.condim = 3

        # plane (same height as the PyBullet plane)
        plane = spec.worldbody.add_geom(name="ground", type=mj.mjtGeom.mjGEOM_PLANE,
                                        pos=[0, 0, params.ground_height], size=[1, 1, 0.1],
                                        rgba=[0.8, 0.8, 0.8, 1])
        plane.contype, plane.conaffinity = WORLD, WORLD
        plane.priority = 1
        plane.friction = [mu_e, 0.005, 0.0001]
        plane.condim = 3

        # box: mass / inertia from the same source as the planner (params)
        box = spec.worldbody.add_body(name="box", pos=list(box_pos), quat=list(_xyzw_to_wxyz(box_orn)))
        box.add_freejoint(name="box_joint")
        box.explicitinertial = True
        box.mass = float(params.mass)
        box.ipos = [0, 0, 0]
        box.inertia = [float(x) for x in np.diag(np.asarray(params.inertia))]
        bg = box.add_geom(name="box_geom", type=mj.mjtGeom.mjGEOM_BOX,
                          size=[params.box.length / 2] * 3, rgba=[0, 0, 0, 0.5])
        bg.contype, bg.conaffinity = WORLD, WORLD | FINGER
        bg.priority = 0
        bg.friction = [mu_c, 0.005, 0.0001]
        bg.condim = 3
        self.model = spec.compile()
        self.data = mj.MjData(self.model)
        self.box_geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "box_geom")
        box_joint = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "box_joint")
        self.box_qadr = self.model.jnt_qposadr[box_joint]
        self.box_dadr = self.model.jnt_dofadr[box_joint]

        # ---------------------------------------------------------------- fingers
        self.finger0 = Finger(self, configs[0], prefixes[0])
        self.finger1 = Finger(self, configs[1], prefixes[1])
        self.fingers = [self.finger0, self.finger1]
        self.ee0_id = self.finger0.end_eff_ids[0]
        self.ee1_id = self.finger1.end_eff_ids[0]
        self.ee_ids = [self.ee0_id, self.ee1_id]

        # collision geoms of the moving links (upper / middle / lower), as in the Bullet detector
        for finger, prefix, i in zip(self.fingers, prefixes, (0, 1)):
            for link in ("upper", "middle", "lower"):
                body = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY,
                                     f"{prefix}finger{i}_{link}_link")
                finger.geom_ids += [g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == body]

        dq = np.zeros(self.finger0.nv)
        q = np.array([0., np.pi / 3, -(np.pi / 2 + np.pi / 3)])
        self.q_default = q

        if ee_init is None:
            self.finger0.reset_state(q, dq)
            self.finger1.reset_state(q, dq)
        else:
            ee0_init, ee1_init = ee_init
            q0, _ = pin_ik(self.finger0.pin_robot, ee0_init, self.ee0_id, q_null=q)
            q1, _ = pin_ik(self.finger1.pin_robot, ee1_init, self.ee1_id, q_null=q)
            self.finger0.reset_state(q0, dq)
            self.finger1.reset_state(q1, dq)

        self.set_box_pose(np.concatenate([box_pos, box_orn]))
        mj.mj_forward(self.model, self.data)
        self.collision_detectors = [CollisionDetector(self, f) for f in self.fingers]

        # ---------------------------------------------------------------- viewer / video
        self.viewer = None
        self._n_steps = 0
        self._sync_every = max(1, int(round(1.0 / (60.0 * dt))))  # redraw at ~60 Hz
        self._frames = None
        self._renderer = None
        if render:
            import mujoco.viewer as mj_viewer  # (as-form: keeps `mujoco` a module-level name)
            self.viewer = mj_viewer.launch_passive(self.model, self.data)
            self.viewer.cam.distance = 0.65
            self.viewer.cam.azimuth = 30
            self.viewer.cam.elevation = -45
            self.viewer.cam.lookat[:] = [0, 0, 0]
            self.viewer.sync()

    def add_visual_frame(self, pos, orn, length=0.1, width=1):
        """Draw a coordinate frame in the viewer (no effect without a viewer)."""
        if self.viewer is None:
            return
        axes = [(np.array([length, 0, 0]), [1, 0, 0, 1]),
                (np.array([0, length, 0]), [0, 1, 0, 1]),
                (np.array([0, 0, length]), [0, 0, 1, 1])]
        scn = self.viewer.user_scn
        for vec, rgba in axes:
            if scn.ngeom >= scn.maxgeom:
                return
            g = scn.geoms[scn.ngeom]
            mj.mjv_initGeom(g, mj.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9),
                            np.array(rgba, dtype=np.float32))
            mj.mjv_connector(g, mj.mjtGeom.mjGEOM_CAPSULE, 0.001 * width,
                             np.asarray(pos, dtype=float), np.asarray(pos, dtype=float) + orn @ vec)
            scn.ngeom += 1
        self.viewer.sync()

    def step(self, sleep_factor=0):
        """Integrates the simulation one step forward.

        sleep_factor: sleep `sleep_factor * dt` per step (1 = real time) when a viewer is open.
        """
        for f in self.fingers:
            self.data.qfrc_applied[f.dadr] = f.tau
        if self.viewer is not None and not self.viewer.is_running():
            self.viewer = None  # window was closed: continue headless
        if self.viewer is not None:
            with self.viewer.lock():
                mj.mj_step(self.model, self.data)
            self._n_steps += 1
            if self._n_steps % self._sync_every == 0:
                self.viewer.sync()
            if sleep_factor:
                time.sleep(sleep_factor * self.dt)
        else:
            mj.mj_step(self.model, self.data)
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
        mj.mj_kinematics(self.model, self.data)

    def get_box_pose(self):
        a = self.box_qadr
        pos = self.data.qpos[a:a + 3].copy()
        quat = _wxyz_to_xyzw(self.data.qpos[a + 3:a + 7])
        return pin.XYZQUATToSE3(np.concatenate([pos, quat]))

    def start_video_recording(self, file_name):
        """Record frames (30 fps) during step(); saved as mp4 by stop_video_recording (needs imageio)."""
        self._video_file = file_name
        self._renderer = mj.Renderer(self.model, height=480, width=640)
        self._frames = []

    def stop_video_recording(self):
        if self._frames is None:
            return
        import imageio
        imageio.mimsave(self._video_file, self._frames, fps=30)
        self._renderer.close()
        self._frames, self._renderer = None, None
