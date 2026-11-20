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

from cto.objects import Cube
from cto.controllers import ImpedanceController
from cto.envs.fingers import FingerDoubleAndBox
from cto.trajectory import generate_ee_motion
from cto.mcts.pvmcts import PolicyValueMCTS
from cto.params import get_default_params, update_params
from cto.contact_modes import construct_contact_plan
from robot_properties_nyu_finger.config import NYUFingerDoubleConfig0, NYUFingerDoubleConfig1

# ## Set up and solve the problem


object_urdf = str(project_path / 'cto' / 'envs' / 'resources' / 'box.urdf')
robot_config = [NYUFingerDoubleConfig0(), NYUFingerDoubleConfig1()]
params = get_default_params(object_urdf, robot_config)

# slide
z = params.box_com_height
desired_poses = [np.array([0, 0.0, z, 0, 0, 0]),
                 np.array([0, 0.1, z, 0, 0, 0])]
params = update_params(params, desired_poses)

# # lift
# z = params.box_com_height
# desired_poses = [np.array([0, 0.0, z, 0, 0, 0]),
#                  np.array([0, 0.0, z + 0.1, 0, 0, 0])]
# params = update_params(params, desired_poses)


# # rotate the cube twice
# z = params.box_com_height
# desired_poses = [np.array([0, 0.0, z, 0, 0, 0]), 
#                  np.array([0, 0.0, z, 0, 0, np.pi/2]),
#                  np.array([0, 0.0, z, 0, 0, np.pi])]
# params = update_params(params, desired_poses)


# # slide with curvature twice
# z = params.box_com_height
# desired_poses = [np.array([-0.05, 0., z, 0, 0, 0]), 
#                  np.array([0.05, 0., z, 0, 0, np.pi/3]),
#                  np.array([0.05, -0.05, z, 0, 0, np.pi])]
# params = update_params(params, desired_poses)


# # pivot about the y-axis by 45 degree
# z = params.box_com_height
# p = 0.05
# rot = 30 * np.pi/ 180
# th = rot + 45 * np.pi / 180
# dx = p - np.cos(th) * np.sqrt(2) * p
# dz = np.sin(th) * np.sqrt(2) * p - p

# desired_poses = [np.array([0,  0., z     , 0, 0, 0]),
#                  np.array([dx, 0., z + dz, 0, rot, 0])]
# params = update_params(params, desired_poses)


pose_init = pin.SE3ToXYZQUAT(params.desired_poses[0])
box_pos = pose_init[:3]
box_orn = pose_init[3:]
env = FingerDoubleAndBox(params, box_pos, box_orn, server=pybullet.DIRECT)

max_budget = 200
mcts = PolicyValueMCTS(params, env)
eval = False
if eval:
    mcts.predictor.load_pvnet(f'{project_path}/models/pvnet2')
    mcts.predictor.load_value_classifier(f'{project_path}/models/value_classifier2')
mcts.run(state=[[0, 0]], budget=max_budget, verbose=True)
state, sol = mcts.get_solution()
env.close()

# ## Generate end-effector motion


dt_plan = 0.1
dt_sim = 1e-3
rest_locations, trajs, forces = generate_ee_motion(state, sol, dt_sim, dt_plan, params)

# ## Simulate


ee_pos = [trajs[0][0][0], trajs[0][1][0]]
box_pos = pin.SE3ToXYZQUAT(params.pose_start)[:3]
box_orn = pin.SE3ToXYZQUAT(params.pose_start)[3:]

env = FingerDoubleAndBox(params, box_pos, box_orn, ee_pos, pybullet.GUI)

controller0 = ImpedanceController(np.diag([100] * 3), np.diag([5.] * 3),
                                  env.finger0.pin_robot, env.ee0_id)
controller1 = ImpedanceController(np.diag([100] * 3), np.diag([5.] * 3),
                                  env.finger1.pin_robot, env.ee1_id)

for i in range(1, len(params.desired_poses)):
    pose = params.desired_poses[i]
    env.add_visual_frame(pose.translation, pose.rotation)

ee0_des, ee1_des = trajs[0][0][0], trajs[0][1][0]
# Run the simulator for 2000 steps to move to the initial position
for i in range(2000):
    # update kinematic
    q0, dq0 = env.finger0.get_state_update_pinocchio()
    q1, dq1 = env.finger1.get_state_update_pinocchio()

    # calculate torque
    tau0 = controller0.compute_torque(q0, dq0, ee0_des, np.zeros(3), np.zeros(3))
    tau1 = controller1.compute_torque(q1, dq1, ee1_des, np.zeros(3), np.zeros(3))

    # send torque
    env.finger0.send_joint_command(tau0)
    env.finger1.send_joint_command(tau1)

    # Step the simulator.
    env.step()

d = params.contact_duration
for i in range(len(state)):
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
        q0, dq0 = env.finger0.get_state_update_pinocchio()
        q1, dq1 = env.finger1.get_state_update_pinocchio()

        # calculate torque
        tau0 = controller0.compute_torque(q0, dq0, ee0_des, np.zeros(3), f0_des)
        tau1 = controller1.compute_torque(q1, dq1, ee1_des, np.zeros(3), f1_des)

        # send torque
        env.finger0.send_joint_command(tau0)
        env.finger1.send_joint_command(tau1)

        # Step the simulator.
        env.step(1)

print("final box pose:", env.get_box_pose(), " target poses:", params.desired_poses)
env.close()
