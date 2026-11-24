"""config

Store the configuration of the robots.

License: BSD 3-Clause License
Copyright (C) 2018-2021, New York University , Max Planck Gesellschaft
Copyright note valid unless otherwise stated in individual files.
All rights reserved.
"""

from dataclasses import dataclass
from math import pi as math_pi
import numpy as np
from cto.envs.multi_articulators import ArticulatorConfig


@dataclass
class NYUFingerConfig(ArticulatorConfig):
    robot_family = "nyu_finger"
    robot_name = "nyu_finger"

    # Initial state
    q_home = np.array([0., np.pi / 3, -(np.pi / 2 + np.pi / 3)])
    dq_home = [0., 0., 0.]

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

    # Maximum control one can send, here the control is the current
    max_control = max_current
    max_qref = math_pi
    ctrl_manager_current_to_control_gain = 1.0

    # Joints
    joint_names = ["finger_base_to_upper_joint",
                   "finger_upper_to_middle_joint",
                   "finger_middle_to_lower_joint"]

    # End effectors
    ee_names = ['finger_tip_link']

    def __post_init__(self):
        super().__post_init__()
        self.pin_robot = self.build_pin_robot()
        self.init_pin_robot()
        self.joints_num = self.pin_robot_model.nv - 6


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

    # Initial state
    q_home = 6 * [0.]
    dq_home = q_home

    def __post_init__(self):
        super().__post_init__()
        self.q0 = np.zeros(self.pin_robot_model.nq)
        self.q0[:] = self.q_home


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
