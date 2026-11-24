#!/usr/bin/env python
# coding: utf-8

import os
import pathlib

project_path = pathlib.Path(__file__).resolve().parent.parent
os.sys.path.insert(1, str(project_path))

from dotmap import DotMap

import numpy as np
import matplotlib.pyplot as plt
import pinocchio as pin

from cto.objects import Cube
from cto.controllers import ImpedanceController
from cto.envs.fingers import FingerDoubleAndBox
from cto.trajectory import generate_ee_motion
from cto.mcts.pvmcts import PolicyValueMCTS
from cto.params import get_default_params, update_params
from cto.kinematics import _xyzw_to_wxyz
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


box_pose = pin.SE3ToXYZQUAT(params.desired_poses[0])
box_pos = box_pose[:3]
box_quat = _xyzw_to_wxyz(box_pose[3:])
box_pose = np.concat([box_pos, box_quat])
env = FingerDoubleAndBox(params, box_pose)

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
env = FingerDoubleAndBox(params, box_pose, ee_pos, render=True)

controller0 = ImpedanceController(np.diag([100] * 3), np.diag([5.] * 3),
                                  env.art0.pin_robot, env.art0.main_ee_id)
controller1 = ImpedanceController(np.diag([100] * 3), np.diag([5.] * 3),
                                  env.art1.pin_robot, env.art1.main_ee_id)

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

face_mats = env.box_face_materials
# T = params.contact_duration
for i in range(len(state)):
    traj0, traj1 = trajs[i]
    force0, force1 = forces[i]
    N0 = len(traj0)
    N1 = len(traj1)
    for n in range(np.max((N0, N1))):
        face0_id, face1_id = np.array(state[i]) % len(face_mats)
        face0_geom = env.model.geom(env.box_face_names[face0_id])
        face1_geom = env.model.geom(env.box_face_names[face1_id])
        face0_geom.matid = face_mats[face0_id]
        face1_geom.matid = face_mats[face1_id]
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
    # face0_geom.matid = face1_geom.matid = env.mat_core
    # env.step(1)

final_pose = env.get_box_pose_se3()
print("final box position:", final_pose.translation, " target:", params.pose_end.translation)
print("final box rotation error [rad]:", np.linalg.norm(pin.log3(final_pose.rotation.T @ params.pose_end.rotation)))
env.close()
