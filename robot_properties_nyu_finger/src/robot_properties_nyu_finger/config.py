"""config

Store the configuration of the Solo family robots.

License: BSD 3-Clause License
Copyright (C) 2018-2021, New York University , Max Planck Gesellschaft
Copyright note valid unless otherwise stated in individual files.
All rights reserved.
"""
from os import path
from dataclasses import dataclass
import numpy as np
from math import pi as math_pi
from pinocchio.utils import zero
from pinocchio.robot_wrapper import RobotWrapper
from robot_properties_nyu_finger.utils import find_paths


@dataclass
class NYUFingerConfig:
    robot_family = "nyu_finger"
    robot_name = "nyu_finger"
    urdf_path: str = ""
    mjcf_path: str = ""
    meshes_path: str = ""

    # PID gains
    kp = 5.0
    kd = 0.1
    ki = 0.0

    # The Kt constant of the motor [Nm/A]: tau = I * Kt.
    motor_torque_constant = 0.025

    # The inertia of a single blmc_motor
    motor_inertia = 0.0000045

    # The motor gear ratio.
    motor_gear_ration = 9.0

    # Control time period.
    control_period = 0.001
    dt = control_period

    # MaxCurrent = 12 # Ampers
    max_current = 2

    # Maximum torques.
    max_torque = motor_torque_constant * max_current

    # Maximum control one can send, here the control is the current.
    max_control = max_current
    max_qref = math_pi

    # ctrl_manager_current_to_control_gain I am not sure what it does so 1.0.
    ctrl_manager_current_to_control_gain = 1.0

    # Joints
    joint_names = ["finger_base_to_upper_joint",
                   "finger_upper_to_middle_joint",
                   "finger_middle_to_lower_joint"]

    # Define the initial state.
    initial_configuration = [0., 0., 0.]
    initial_velocity = [0., 0., 0.]

    # End effectors
    ee_names = ['finger_tip_link']

    def build_pin_robot_wrapper(self):
        # Rebuild the robot wrapper instead of using the existing model to also load the visuals
        # NOTE: Pinocchio still favors URDF over MJCF
        robot = RobotWrapper.BuildFromURDF(self.desc_path, self.meshes_path) if self.desc_path.endswith(".urdf") \
            else RobotWrapper.BuildFromMJCF(self.desc_path)
        robot.model.rotorInertia[6:] = self.motor_inertia
        robot.model.rotorGearRatio[6:] = self.motor_gear_ration
        return robot

    def joint_name_in_single_string(self):
        joint_names = ""
        for name in self.pin_robot_model.names[2:]:
            joint_names += name + " "
        return joint_names

    def __post_init__(self):
        paths = find_paths(self.robot_name)
        self.resources_path = paths["resources"]
        self.meshes_path = paths["meshes"]
        self.dgm_yaml_dir = paths["dgm_yaml_dir"]
        self.dgm_yaml_path = paths["dgm_yaml"]
        self.urdf_path = paths["urdf"]
        self.mjcf_path = paths["mjcf"]
        self.desc_path = self.mjcf_path

        # pinocchio model
        self.pin_robot = self.build_pin_robot_wrapper()
        self.pin_robot_model = self.pin_robot.model
        self.mass = np.sum([i.mass for i in self.pin_robot_model.inertias])

        # End effectors info
        self.ee_ids = [self.pin_robot_model.getFrameId(ee_frame_name) for ee_frame_name in self.ee_names]
        self.ees_num = len(self.ee_names)

        # The number of motors, here they are the same as there are only revolute joints
        self.joints_num = self.pin_robot_model.nv - 6

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

        self.q0 = zero(self.pin_robot_model.nq)
        self.q0[:] = self.initial_configuration
        self.v0 = zero(self.pin_robot_model.nv)
        self.a0 = zero(self.pin_robot_model.nv)


@dataclass
class NYUFingerDoubleConfig0(NYUFingerConfig):
    robot_name = "nyu_finger_double0"

    # End effectors info
    ee_names = ['finger0_tip_link']

    # The number of motors, here they are the same as there are only revolute joints
    joint_names = ["finger0_base_to_upper_joint",
                   "finger0_upper_to_middle_joint",
                   "finger0_middle_to_lower_joint"]


@dataclass
class NYUFingerDoubleConfig1(NYUFingerConfig):
    robot_name = "nyu_finger_double1"

    # The number of motors, here they are the same as there are only revolute joints
    joint_names = ["finger1_base_to_upper_joint",
                   "finger1_upper_to_middle_joint",
                   "finger1_middle_to_lower_joint"]

    # End effectors
    ee_names = ['finger1_tip_link']


@dataclass
class NYUFingerDoubleConfig(NYUFingerConfig):
    robot_name = "nyu_finger_double"

    # End effectors info
    ee_names = ['finger0_tip_link', 'finger1_tip_link']

    # Joints
    joint_names = ["finger0_base_to_upper_joint",
                   "finger0_upper_to_middle_joint",
                   "finger0_middle_to_lower_joint",
                   "finger1_base_to_upper_joint",
                   "finger1_upper_to_middle_joint",
                   "finger1_middle_to_lower_joint"]

    # Define the initial state.
    initial_configuration = [0., 0., 0., 0., 0., 0.]
    initial_velocity = [0., 0., 0., 0., 0., 0.]

    def __post_init__(self):
        super().__post_init__()
        self.q0 = zero(self.pin_robot_model.nq)
        self.q0[:] = self.initial_configuration


@dataclass
class NYUFingerTripleConfig0(NYUFingerConfig):
    robot_name = "nyu_finger_triple0"

    # End effectors
    ee_names = ['finger0_tip_link']

    joint_names = ["finger0_base_to_upper_joint",
                   "finger0_upper_to_middle_joint",
                   "finger0_middle_to_lower_joint"]

    def __post_init__(self):
        super().__post_init__()
        # The number of motors, here they are the same as there are only revolute joints
        self.nb_joints = self.pin_robot_model.nv


@dataclass
class NYUFingerTripleConfig1(NYUFingerConfig):
    robot_name = "nyu_finger_triple1"

    # End effectors
    ee_names = ['finger1_tip_link']

    joint_names = ["finger1_base_to_upper_joint",
                   "finger1_upper_to_middle_joint",
                   "finger1_middle_to_lower_joint"]

    def __post_init__(self):
        super().__post_init__()
        # The number of motors, here they are the same as there are only revolute joints
        self.nb_joints = self.pin_robot_model.nv


@dataclass
class NYUFingerTripleConfig2(NYUFingerConfig):
    robot_name = "nyu_finger_triple2"

    # End effectors info
    ee_names = ['finger2_tip_link']

    # Joints
    joint_names = ["finger2_base_to_upper_joint",
                   "finger2_upper_to_middle_joint",
                   "finger2_middle_to_lower_joint"]

    def __post_init__(self):
        super().__post_init__()
        # The number of motors, here they are the same as there are only revolute joints
        self.nb_joints = self.pin_robot_model.nv
