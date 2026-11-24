#!/usr/bin/env python
# coding: utf-8

import os
import pathlib

project_path = pathlib.Path(__file__).resolve().parent.parent
os.sys.path.insert(1, str(project_path))

from dotmap import DotMap
import pybullet

import numpy as np
import matplotlib.pyplot as plt
import pinocchio as pin
import pybullet

import gurobipy as gp
from gurobipy import GRB

from cto.objects import Cube
from cto.controllers import ImpedanceController
from cto.envs.fingers import FingerDoubleAndBox
from cto.trajectory import generate_ee_motion
from cto.miqp.problems import MIQP
from cto.params import get_default_params, update_params
from cto.contact_modes import construct_contact_plan
from robot_properties_nyu_finger.config import NYUFingerDoubleConfig0, NYUFingerDoubleConfig1

# ## Set up and solve the problem


object_urdf = str(project_path / 'cto' / 'envs' / 'resources' / 'box.urdf')
robot_config = [NYUFingerDoubleConfig0(), NYUFingerDoubleConfig1()]
params = get_default_params(object_urdf, robot_config, MIQP=True)
params.contact_duration = 1  # long contact duration slows down MIQP significantly

# slide the cube
z = params.box_com_height
desired_poses = [np.array([0, 0.0, z, 0, 0, 0]),
                 np.array([0, 0.05, z, 0, 0, 0])]
params = update_params(params, desired_poses)

# # rotate the cube
# z = params.box_com_height
# desired_poses = [np.array([0, 0.0, z, 0, 0, 0]), 
#                  np.array([0, 0.0, z, 0, 0, np.pi/4])]
# params = update_params(params, desired_poses)


# # lift the cube
# z = params.box_com_height
# desired_poses = [np.array([0, 0.0, z, 0, 0, 0]), 
#                  np.array([0, 0.0, z + 0.1, 0, 0, 0])]
# params = update_params(params, desired_poses)


miqp = MIQP(params)
miqp.setup()
sol = miqp.solve(nsol=20)  # explore more feasible solutions to reduce force error

# ## Generate end-effector motion


dt_plan = 0.1
dt_sim = 1e-3
rest_locations, trajs, forces = generate_ee_motion(sol.contact_modes, sol, dt_sim, dt_plan, params)

# ## Simulate


ee_pos = [trajs[0][0][0], trajs[0][1][0]]
box_pos = pin.SE3ToXYZQUAT(params.pose_start)[:3]
box_orn = pin.SE3ToXYZQUAT(params.pose_start)[3:]

env = FingerDoubleAndBox(params, box_pos, box_orn, ee_pos, pybullet.GUI)

controller0 = ImpedanceController(np.diag([180] * 3), np.diag([5.] * 3),
                                  env.art0.pin_robot, env.ee0_id)
controller1 = ImpedanceController(np.diag([180] * 3), np.diag([5.] * 3),
                                  env.art1.pin_robot, env.ee1_id)

for i in range(1, len(params.desired_poses)):
    pose = params.desired_poses[i]
    env.add_visual_frame(pose.translation, pose.rotation)

ee0_des, ee1_des = trajs[0][0][0], trajs[0][1][0]
# Run the simulator for 2000 steps to move to the initial position
for i in range(2000):
    # update kinematic
    q0, dq0 = env.art0.get_state_update_pinocchio()
    q1, dq1 = env.art1.get_state_update_pinocchio()

    # calculate torque
    tau0 = controller0.compute_torque(q0, dq0, ee0_des, np.zeros(3), np.zeros(3))
    tau1 = controller1.compute_torque(q1, dq1, ee1_des, np.zeros(3), np.zeros(3))

    # send torque
    env.art0.send_joint_command(tau0)
    env.art1.send_joint_command(tau1)

    # Step the simulator.
    env.step()

d = params.contact_duration
for i in range(len(sol.contact_modes)):
    traj0, traj1 = trajs[i]
    force0, force1 = forces[i]
    N0 = len(traj0)
    N1 = len(traj1)
    for n in range(np.max((N0, N1))):
        n0 = n if n < N0 else -1
        n1 = n if n < N1 else -1
        ee0_des = traj0[n0]
        ee1_des = traj1[n1]

        f0_des = force0[n0]
        f1_des = force1[n1]

        # update kinematic
        q0, dq0 = env.art0.get_state_update_pinocchio()
        q1, dq1 = env.art1.get_state_update_pinocchio()

        # calculate torque
        tau0 = controller0.compute_torque(q0, dq0, ee0_des, np.zeros(3), f0_des)
        tau1 = controller1.compute_torque(q1, dq1, ee1_des, np.zeros(3), f1_des)

        # send torque
        env.art0.send_joint_command(tau0)
        env.art1.send_joint_command(tau1)

        # Step the simulator.
        env.step(1)

env.close()
