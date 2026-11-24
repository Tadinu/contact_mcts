"""
Bimanual Franka Panda + box in MuJoCo; same interface as cto/envs/fingers_mj.FingerDoubleAndBox.

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
collision_detectors, set_box_pose, get_box_pose, add_visual_frame, step, close, start/stop_video_recording.

Joint-limit violations of an IK result are reported by collision_detectors[c].in_collision as "in collision",
because cto/kinematics.inverse_kinematics_3d does not enforce limits.

Friction is explicit (geom priority): arm tool/links <-> box = params.friction_coeff,
box <-> table = params.environment_friction, true elliptic friction cone.
Gravity of the arms is compensated with MuJoCo's bias forces (the arms are heavy, unlike the fingers).
"""
from __future__ import annotations
from dataclasses import dataclass
import os
import numpy as np
import mujoco as mj
from pinocchio.robot_wrapper import RobotWrapper
from cto.envs.multi_articulators import ArticulatorConfig, Articulator, MultiArticulatorEnv
from cto.kinematics import pin_model_offset_link

PANDA_XML_DIR = "/home/tad/1_MUJOCO/MJ_GRASP/mjmanip/mjmanip/models/panda"
PANDA_PALM_NOHAND_XML = f"{PANDA_XML_DIR}/panda_palm_nohand.xml"
PANDA_NOHAND_XML = f"{PANDA_XML_DIR}/panda_nohand.xml"
DEFAULT_PANDA_XML = PANDA_PALM_NOHAND_XML


@dataclass
class PandaConfig(ArticulatorConfig):
    robot_family = "panda"
    robot_name = "panda_nohand"
    mjcf_path = DEFAULT_PANDA_XML
    meshes_path = ""
    desc_path = mjcf_path

    base_distance = 0.70

    # Joints
    joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
    joint_damping = 0.1
    joint_armature = 0.02

    # End effectors
    attachment_name = "attachment"
    tool_ee_name = "ee"
    tool_radius = 0.01
    ee_names = [tool_ee_name]

    # Initial state
    q_home = np.array([0.0, 0.318, 0.0, -2.224, 0.0, 2.541, 0.785])


@dataclass
class PandaWithToolConfig(PandaConfig):
    # End effectors
    tool_ee_name = "tool_flange"
    tool_length = 0.2
    tool_radius = 0.01
    ee_names = [tool_ee_name]


class Panda(Articulator):
    pass


class BimanualPandaAndBox(MultiArticulatorEnv):
    def __init__(self, params, box_pose, ee_init=None, render=False, dt=0.001):
        config_cls = PandaWithToolConfig if DEFAULT_PANDA_XML is PANDA_NOHAND_XML else PandaConfig
        super().__init__(articulator_class=Panda, params=params,
                         configs=[config_cls(idx=0), config_cls(idx=1)],
                         box_pose=box_pose,
                         ee_init=ee_init, render=render, dt=dt)

        for i, prefix in enumerate(self.prefixes):
            art = self.articulators[i]
            for j in range(1, 8):
                link_j = self.scene_spec.body(f"{prefix}link{j}")
                art.geom_ids.extend([g.id for g in link_j.geoms if g.contype != 0])
            art.geom_ids.extend([g.id for g in self.scene_spec.body(f"{prefix}{art.config.ee_names[0]}").geoms
                                 if g.contype != 0])

    def init_articulators(self):
        # Articulators (pinocchio models share the same MJCF with mujoco ones)
        for art in self.articulators:
            # Rebuild arm's [pin_robot] if needed
            cfg = art.config
            if cfg.desc_path != cfg.mjcf_path:
                cfg.desc_path = cfg.mjcf_path

            # Build robot
            cfg.pin_robot = cfg.build_pin_robot()  # from [desc_path]

            # NOTE: Pinnochio does not read MJ's base pose setting, so it must manually offset here following MJ
            pin_robot_model = cfg.pin_robot.model
            pin_model_offset_link(pin_robot_model, 1, cfg.base_pose(self.params.ground_height))

            # Reset [pin_robot]'s data
            cfg.pin_robot = RobotWrapper(pin_robot_model)

            # Init [pin_robot]
            cfg.init_pin_robot()

    def init_mujoco_articulator_spec(self, art_spec: mj.MjSpec, config: ArticulatorConfig):
        super().init_mujoco_articulator_spec(art_spec, config)
        # No need for actuators, torque calculated from [pinocchio dynamics] are applied directly to [qfrc_applied]
        for act in list(art_spec.actuators):
            art_spec.delete(act)
        for k in list(art_spec.keys):
            art_spec.delete(k)
        for j in art_spec.joints:
            j.damping = [config.joint_damping, 0.0, 0.0]  # (array field in MjSpec; hinge joints use [0])
            j.armature = config.joint_armature

        # Enable all bodies' [gravcomp]
        for body in art_spec.bodies:
            body.gravcomp = 1

        # Move base
        art_base_pose = config.base_pose(self.params.ground_height)
        art_base = art_spec.bodies[1]
        art_base.pos = art_base_pose[:3]
        art_base.quat = art_base_pose[3:]

        # Add tool
        attachment = art_spec.body(PandaConfig.attachment_name)
        if isinstance(config, PandaWithToolConfig):
            # Vertical tool (pointing straight down) hovering at (-/+0.15, 0, 0.14)
            # -> Its shaft leans over the box edge and pinches it there instead of the sphere touching the face.
            # NOTE: For consistency, Both MuJoCo and Pinocchio must share the exactly same XML!
            shaft = attachment.add_geom(type=mj.mjtGeom.mjGEOM_CAPSULE,
                                        fromto=[0, 0, 0, 0, 0, config.tool_length],
                                        size=[0.006, 0, 0], rgba=[0.3, 0.3, 0.3, 1])
            tool_ee = attachment.add_body(name=config.tool_ee_name, pos=[0, 0, config.tool_length])
            tool_ee.gravcomp = 1
            tip = tool_ee.add_geom(name=config.tool_ee_name, type=mj.mjtGeom.mjGEOM_SPHERE,
                                   size=[config.tool_radius, 0, 0], rgba=[0.9, 0.2, 0.2, 1])
            for g in (shaft, tip):
                g.mass = 0.01 if g is tip else 0.05
                self.configure_articulator_geom_collision(g)
        else:
            palm = art_spec.geom("palm")
            palm_ee = attachment.add_body(name=config.tool_ee_name, pos=[0, 0, 0.15])
            palm_ee.gravcomp = 1
            tip = palm_ee.add_geom(name=palm_ee.name, type=mj.mjtGeom.mjGEOM_SPHERE,
                                   size=[config.tool_radius, 0, 0], rgba=[0.9, 0.2, 0.2, 1])
            for g in (palm, tip):
                g.mass = 0.01
                self.configure_articulator_geom_collision(g)

        # Save [art_spec] to file (must be done through its copy as [art_spec] is used to attach to scene later)
        art_spec.meshdir = str(os.path.join(art_spec.modelfiledir, art_spec.meshdir))
        # NOTE: Saving [art_spec] to file requires prior spec compiling into model!
        art_spec_copy = art_spec.copy()
        art_spec_copy.compile()
        art_spec_copy.to_file(f"{art_spec.modelname}.xml")
        # Update [mjcf_path] to the newly saved one on disk
        config.mjcf_path = f"{art_spec.modelname}.xml"
