"""
Differences from the PyBullet environment (deliberate):
  * Friction is explicit. MuJoCo normally takes the MAX of the two geoms' coefficients (Bullet
    multiplies them), so we use geom `priority` to make the contact use exactly
        finger <-> box : params.friction_coeff
        box    <-> plane: params.environment_friction
    with a true (elliptic) Coulomb cone, like pybullet's enableConeFriction=1.
  * Joint torques are applied directly with qfrc_applied, unclipped
"""
from typing import Optional
from dataclasses import dataclass, field
import numpy as np

import time
import tempfile

import numpy as np
import numpy.typing as npt
import mujoco as mj
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

from robot_properties_nyu_finger.utils import find_paths
from cto.kinematics import pin_ik
from cto.envs.collision import CollisionDetector


def mj_spec_scale_body(body: mj.MjsBody, scale: npt.ArrayLike) -> None:
    """Recursively scale a body and all its child geoms, sites, joints, and sub-bodies."""
    # 1- Geoms (sizes and local offsets)
    for geom in body.geoms:
        geom.pos = np.array(geom.pos) * scale
        geom.size = np.array(geom.size) * scale

        # If using fromto (capsules, cylinders, boxes defined by endpoints)
        if not np.allclose(geom.fromto, 0):
            geom.fromto = np.array(geom.fromto) * scale

    # 2- Sites
    for site in body.sites:
        site.pos = np.array(site.pos) * scale
        site.size = np.array(site.size) * scale

    # 3- Joints (local anchor positions)
    for joint in body.joints:
        joint.pos = np.array(joint.pos) * scale

    # 4- Recursively scale child bodies
    for child in body.bodies:
        child.pos = np.array(child.pos) * scale
        mj_spec_scale_body(child, scale)


@dataclass
class ArticulatorConfig:
    idx: int = field(default=0, kw_only=True)
    robot_family = ""
    robot_name = ""
    urdf_path = ""
    mjcf_path = ""
    desc_path = ""
    meshes_path = ""

    # Base pose
    base_distance = None

    def base_pose(self, z: float) -> np.ndarray:
        if self.base_distance is not None:
            base_xyz, base_yaw = ([-self.base_distance, 0.0, z], 0.0) if self.idx == 0 \
                else ([self.base_distance, 0.0, z], np.pi)
            base_xyz = np.array(base_xyz)
            base_quat = np.zeros(4)
            mj.mju_euler2Quat(base_quat, [0.0, 0.0, base_yaw], "XYZ")
            return np.concat([base_xyz, base_quat])
        else:
            return np.array([0, 0, z, 1, 0, 0, 0])

    # The inertia of a single motor
    motor_inertia = None

    # The motor gear ratio.
    motor_gear_ration = None

    # Initial state
    q_home = None
    dq_home = None

    # End effectors
    ee_names = []

    def __post_init__(self):
        if not self.desc_path:
            paths = find_paths(self.robot_name)
            self.resources_path = paths["resources"]
            self.meshes_path = paths["meshes"]
            self.dgm_yaml_dir = paths["dgm_yaml_dir"]
            self.dgm_yaml_path = paths["dgm_yaml"]
            self.urdf_path = paths["urdf"]
            self.mjcf_path = paths["mjcf"]
            self.desc_path = self.mjcf_path

        # Pinocchio robot (up to child classes on when to initialize)
        self.pin_robot: pin.RobotWrapper = None
        self.pin_robot_model: pin.Model = None

    def init_pin_robot(self):
        # Reinit in case [pin_robot] having been rebuilt on the fly
        self.pin_robot_model = self.pin_robot.model
        if self.motor_inertia:
            self.pin_robot_model.rotorInertia[6:] = self.motor_inertia
        if self.motor_gear_ration:
            self.pin_robot_model.rotorGearRatio[6:] = self.motor_gear_ration
        self.mass = np.sum([i.mass for i in self.pin_robot_model.inertias])

        # End effectors info
        self.ee_ids = [self.pin_robot_model.getFrameId(ee_name) for ee_name in self.ee_names]

        # Mapping between the ctrl vector in the device and the mjcf indexes.
        self.xml_to_dgm = tuple(range(3))

        self.map_joint_name_to_id = {}
        self.map_joint_limits = {}
        for i, (name, lb, ub) in enumerate(
                zip(
                    self.pin_robot_model.names[1:],
                    self.pin_robot_model.lowerPositionLimit,
                    self.pin_robot_model.upperPositionLimit,
                )
        ):
            self.map_joint_name_to_id[name] = i
            self.map_joint_limits[i] = [float(lb), float(ub)]

        self.q0 = np.zeros(self.pin_robot_model.nq)
        if self.q_home is not None:
            self.q0[:] = self.q_home
        self.v0 = np.zeros(self.pin_robot_model.nv)
        self.a0 = np.zeros_like(self.v0)

    def build_pin_robot(self) -> pin.RobotWrapper:
        return RobotWrapper.BuildFromURDF(self.desc_path, self.meshes_path) if self.desc_path.endswith(".urdf") \
            else RobotWrapper.BuildFromMJCF(self.desc_path)

    def joint_name_in_single_string(self):
        joint_names = ""
        for name in self.pin_robot_model.names[2:]:
            joint_names += f"{name}, "
        return joint_names


class Articulator:
    Q_HOME = []

    def __init__(self, env, config: ArticulatorConfig, prefix: str):
        self.env = env
        self.config = config
        model = env.model
        jids = [model.joint(f"{prefix}{n}").id for n in config.joint_names] if config.joint_names else (
            list(range(env.model.njnt)))
        assert all(j >= 0 for j in jids), "articulator joints not found in the MuJoCo model"
        self.qadr = model.jnt_qposadr[jids]
        self.dadr = model.jnt_dofadr[jids]
        self.tau = np.zeros(len(jids))
        self.geom_ids = []  # collision geoms of the moving links
        self.prefix = prefix

    @property
    def pin_robot(self) -> pin.RobotWrapper:
        return self.config.pin_robot

    @property
    def pin_model(self) -> pin.Model:
        return self.pin_robot.model

    @property
    def main_ee_id(self) -> int:
        return self.config.ee_ids[0]

    @property
    def nv(self) -> int:
        return self.pin_model.nv

    @property
    def nq(self) -> int:
        return self.pin_model.nq

    @property
    def lower(self) -> int:
        return self.pin_model.lowerPositionLimit.copy()

    @property
    def upper(self) -> int:
        return self.pin_model.upperPositionLimit.copy()

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


class MultiArticulatorEnv:
    MODELS_DIR = "/home/tad/1_MUJOCO/MJ_GRASP/mjmanip/mjmanip/models"
    ROBOT, WORLD = 2, 1

    def __init__(self, articulator_class: type[Articulator], params, configs: list, box_pose, ee_init=None,
                 render=False,
                 dt=0.001):
        self.articulator_class = articulator_class
        self.dt = dt
        self.params = params
        self.configs = configs
        self.mu_e = params.environment_friction
        self.mu_c = params.friction_coeff

        # Spec: scene
        self.scene_spec = mj.MjSpec()
        self.scene_spec.option.timestep = dt
        self.scene_spec.option.gravity = [0, 0, -9.81]
        self.scene_spec.option.cone = mj.mjtCone.mjCONE_ELLIPTIC  # true Coulomb cone
        self.scene_spec.option.impratio = 10  # stiffer friction

        self._tmp = tempfile.TemporaryDirectory()
        self.prefixes = ["r0_", "r1_"]

        # Spec: robot instance
        for i, prefix in enumerate(self.prefixes):
            cfg = self.configs[i]
            art_i_spec = mj.MjSpec.from_file(cfg.mjcf_path)
            art_i_spec.modelname = f"{prefix}{art_i_spec.modelname}"
            self.init_mujoco_articulator_spec(art_i_spec, cfg)
            self.scene_spec.attach(art_i_spec,
                                   frame=self.scene_spec.worldbody.add_frame(name=art_i_spec.modelname),
                                   prefix=prefix)

        # Spec: Contact bit masks: robots collide with box and plane, never with each other
        for g in self.scene_spec.geoms:
            if g.type == mj.mjtGeom.mjGEOM_MESH and "collision" in g.classname.name:
                self.configure_articulator_geom_collision(g)

        # Spec: plane
        plane = self.scene_spec.worldbody.add_geom(name="ground", type=mj.mjtGeom.mjGEOM_PLANE,
                                                   pos=[0, 0, params.ground_height], size=[1, 1, 0.1],
                                                   rgba=[0.8, 0.8, 0.8, 1])
        plane.contype, plane.conaffinity = self.WORLD, self.WORLD
        plane.priority = 1
        plane.friction = [self.mu_e, 0.005, 0.0001]
        plane.condim = 3

        # Spec: Box
        box = mj.MjSpec.from_file(f"{self.MODELS_DIR}/objects/cube/colorized_cube.xml")
        box_prefix = "/"
        self.scene_spec.attach(box, frame=self.scene_spec.worldbody.add_frame(pos=box_pose[:3], quat=box_pose[3:]),
                               prefix=box_prefix)
        box = self.scene_spec.body(f"{box_prefix}cube")
        box_name = box.name
        # These can only be configured after being attached to [spec]
        box_joint_name = f"{box_name}_freejoint"  # Already having [box_prefix]
        box.add_freejoint(name=box_joint_name)
        box.explicitinertial = True
        box.mass = float(params.mass)
        box.ipos = [0, 0, 0]
        box.inertia = [float(x) for x in np.diag(np.asarray(params.inertia))]
        box_col_geom = [geom for geom in box.geoms if geom.name == box_name][0]
        box_col_geom.contype = self.WORLD
        box_col_geom.conaffinity = self.WORLD | self.ROBOT
        box_col_geom.priority = 0
        box_col_geom.condim = 4
        box_col_geom.friction = [self.mu_c, 0.005, 0.001]

        # Compile scene_spec into model
        self.model = self.scene_spec.compile()
        self.data = mj.MjData(self.model)

        # Model: Box
        self.box_geom_id = self.model.geom(box.name).id
        bj = self.model.joint(box_joint_name).id
        self.box_qadr, self.box_dadr = self.model.jnt_qposadr[bj], self.model.jnt_dofadr[bj]
        self.box_face_names = [f"{box_prefix}cube_visual_face_{i}" for i in range(1, 7)]
        self.box_face_materials = []
        # Initially, dim all faces
        self.mat_core = self.model.mat(f"{box_prefix}mat_core").id
        for face in self.box_face_names:
            self.box_face_materials.append(self.model.geom(face).matid.copy())
            self.model.geom(face).matid = self.mat_core
        self.set_box_pose(box_pose)

        # Model: Articulators
        self.art0 = self.articulator_class(self, configs[0], self.prefixes[0])
        self.art1 = self.articulator_class(self, configs[1], self.prefixes[1])
        self.articulators = [self.art0, self.art1]
        self.init_articulators()
        dq = np.zeros(self.art0.nv)
        if ee_init is None:
            for art in self.articulators:
                art.reset_state(art.config.q_home, dq)
        else:
            for art, ee in zip(self.articulators, ee_init):
                q, _ = pin_ik(art.pin_robot, ee, art.main_ee_id, q_init=art.config.q_home,
                              q_null=art.config.q_home, max_it=300)
                art.reset_state(np.clip(q, art.lower, art.upper), dq)

        # Collision detectors
        self.collision_detectors = [CollisionDetector(self, art) for art in self.articulators]

        # Viewer/renderer
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

    def init_articulators(self):
        pass

    def init_mujoco_articulator_spec(self, art_spec: mj.MjSpec, config: ArticulatorConfig):
        art_spec.compiler.discardvisual = True
        art_spec.compiler.balanceinertia = True

    def configure_articulator_geom_collision(self, g: mj.MjsGeom):
        g.contype, g.conaffinity = self.ROBOT, self.WORLD
        g.priority = 1
        g.friction = [self.mu_c, 0.005, 0.0001]
        g.condim = 3

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
        for art in self.articulators:
            self.data.qfrc_applied[art.dadr] = art.tau
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
        self.data.qpos[a + 3:a + 7] = pose[3:]
        self.data.qvel[self.box_dadr:self.box_dadr + 6] = 0
        mj.mj_kinematics(self.model, self.data)

    def get_box_pose(self):
        a = self.box_qadr
        pos = self.data.qpos[a:a + 3]
        quat = self.data.qpos[a + 3:a + 7]
        return np.concatenate([pos, quat])

    def get_box_pose_se3(self):
        box_pose = self.get_box_pose()
        return pin.XYZQUATToSE3(np.concatenate([box_pose[:3], _wxyz_to_xyzw(box_pose[3:])]))

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
