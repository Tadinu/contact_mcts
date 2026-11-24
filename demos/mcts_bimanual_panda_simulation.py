#!/usr/bin/env python
# coding: utf-8
"""MCTS contact planning + execution with two Franka Panda arms in MuJoCo.

Needs the Panda model from MuJoCo Menagerie in cto/envs/resources/franka_emika_panda/:
    git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie
    cd mujoco_menagerie && git sparse-checkout set franka_emika_panda
    cp -r franka_emika_panda <project>/cto/envs/resources/
"""
import os
import pathlib

project_path = pathlib.Path(__file__).resolve().parent.parent
os.sys.path.insert(1, str(project_path))

import numpy as np
import pinocchio as pin

from cto.controllers import NullspaceImpedanceController
from cto.envs.arms import BimanualPandaAndBox
from cto.trajectory import generate_ee_motion
from cto.mcts.pvmcts import PolicyValueMCTS
from cto.params import get_default_params, update_params
from cto.kinematics import _xyzw_to_wxyz

# ## Set up the problem

# "/home/tad/1_MUJOCO/MJ_GRASP/mjmanip/mjmanip/models/objects/cube/colorized_cube.xml"
object_urdf = str(project_path / 'cto' / 'envs' / 'resources' / 'box.urdf')
params = get_default_params(object_urdf, robot_config=None)

# plan with a friction coefficient no larger than what the robot/simulator really has
# (see the friction discussion: planner mu <= simulator mu worked in every test)
# params.friction_coeff_plan = 0.5

# slide
z = params.box_com_height
desired_poses = [np.array([0, 0.0, z, 0, 0, 0]),
                 np.array([0, 0.2, z + 0.1, 0, 0, 0])]
params = update_params(params, desired_poses)

box_pose = pin.SE3ToXYZQUAT(params.desired_poses[0])
box_pos = box_pose[:3]
box_quat = _xyzw_to_wxyz(box_pose[3:])
box_pose = np.concat([box_pos, box_quat])

## Solve with MCTS
env = BimanualPandaAndBox(params, box_pose)

max_budget = 200
mcts = PolicyValueMCTS(params, env)
mcts.run(state=[[0, 0]], budget=max_budget, verbose=True)
state, sol = mcts.get_solution()
env.close()

## Generate EE motion
dt_plan = 0.1
dt_sim = 1e-3
rest_locations, trajs, forces = generate_ee_motion(state, sol, dt_sim, dt_plan, params)

## Simulate
ee_pos = [trajs[0][0][0], trajs[0][1][0]]
env = BimanualPandaAndBox(params, box_pose, ee_pos, render=True)

# stiff Cartesian impedance + feed-forward force; null-space posture task for the 7th DoF
# kp = np.array([600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0])
# kd = np.array([30.0, 30.0, 30.0, 30.0, 10.0, 10.0, 5.0])
P, D = 600., 60.
controllers = [NullspaceImpedanceController(np.diag([P] * 3), np.diag([D] * 3), art.pin_robot, art.main_ee_id,
                                            q_null=art.config.q_home, Kn=5., Dn=1.)
               for art in env.articulators]

for i in range(1, len(params.desired_poses)):
    pose = params.desired_poses[i]
    env.add_visual_frame(pose.translation, pose.rotation)


def command(x_des, v_des, f_des):
    for art, ctrl, x, v, f in zip(env.articulators, controllers, x_des, v_des, f_des):
        q, dq = art.get_state_update_pinocchio()
        art.send_joint_command(ctrl.compute_torque(q, dq, x, v, f))


# Run the simulator for 2000 steps to move to the initial position
zero = [np.zeros(3), np.zeros(3)]
for i in range(2000):
    command([trajs[0][0][0], trajs[0][1][0]], zero, zero)
    env.step()

face_mats = env.box_face_materials
for i in range(len(state)):
    face0_id, face1_id = np.array(state[i]) % len(face_mats)
    face0_geom = env.model.geom(env.box_face_names[face0_id])
    face1_geom = env.model.geom(env.box_face_names[face1_id])
    face0_geom.matid = face_mats[face0_id]
    face1_geom.matid = face_mats[face1_id]
    traj0, traj1 = trajs[i]
    force0, force1 = forces[i]
    N0 = len(traj0)
    N1 = len(traj1)
    for n in range(np.max((N0, N1))):
        n0 = n if n < N0 else -1
        n1 = n if n < N1 else -1
        # velocity feed-forward from the planned tip trajectory (removes the damping-induced lag)
        v0 = (traj0[min(n0 + 1, N0 - 1)] - traj0[n0]) / dt_sim if n0 >= 0 else np.zeros(3)
        v1 = (traj1[min(n1 + 1, N1 - 1)] - traj1[n1]) / dt_sim if n1 >= 0 else np.zeros(3)

        command([traj0[n0], traj1[n1]], [v0, v1], [force0[n0], force1[n1]])
        env.step(1)
    # face0_geom.matid = face1_geom.matid = env.mat_core
    # env.step(1)

final_pose = env.get_box_pose_se3()
print("final box position:", final_pose.translation, " target:", params.pose_end.translation)
print("final box rotation error [rad]:", np.linalg.norm(pin.log3(final_pose.rotation.T @ params.pose_end.rotation)))
env.close()
