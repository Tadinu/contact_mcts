#!/usr/bin/env python
# coding: utf-8


import os
import pathlib

project_path = pathlib.Path(__file__).resolve().parent.parent
os.sys.path.insert(1, str(project_path))

from dotmap import DotMap
from tqdm.notebook import trange
import torch

import numpy as np
import pinocchio as pin
import pybullet
import matplotlib.pyplot as plt

from cto.objects import Cube
from cto.envs.fingers import FingerDoubleAndBox
from cto.mcts.pvmcts import PolicyValueMCTS
from cto.trajectory import generate_random_poses
from cto.params import get_default_params, update_params
from robot_properties_nyu_finger.config import NYUFingerDoubleConfig0, NYUFingerDoubleConfig1
from cto.exp import random_poses_composite

object_urdf = str(project_path / 'cto' / 'envs' / 'resources' / 'box.urdf')
robot_config = [NYUFingerDoubleConfig0(), NYUFingerDoubleConfig1()]
params = get_default_params(object_urdf, robot_config)

states = []
values = []
action_probs = []
goals = []
eps = 1e-3
failed_tasks = []
all_tasks = []

ntasks = 300
n_desired_poses = 2
max_budget = 200

for i in trange(ntasks):
    if i < 200:
        motion = 'sc'
    elif 200 <= i < 250:
        motion = 'scl'
    elif 250 <= i < 300:
        motion = 'scp'
    desired_poses = random_poses_composite(params, n_desired_poses, motion)
    all_tasks.append(desired_poses)
    params = update_params(params, desired_poses)
    pose_init = pin.SE3ToXYZQUAT(params.desired_poses[0])
    box_pos = pose_init[:3]
    box_orn = pose_init[3:]
    env = FingerDoubleAndBox(params, box_pos, box_orn, server=pybullet.DIRECT)

    mcts = PolicyValueMCTS(params, env)
    mcts.train(state=[[0, 0]], budget=max_budget, verbose=False)
    best_state, _ = mcts.get_solution()

    if best_state is None:
        print('failed')
        failed_tasks.append(desired_poses)
    else:
        states += mcts.get_data()[0]
        values += mcts.get_data()[1]
        action_probs += mcts.get_data()[2]
        goals += mcts.get_data()[3]
    env.close()

from cto.mcts.pvmcts import MCTSDataset

data = MCTSDataset(states, values, action_probs, goals)
torch.save(data, f'{project_path}/data/data')

# visualize the failed tasks on the x-y plane
if len(failed_tasks) > 0:
    plt.scatter(np.vstack(failed_tasks)[:, 0], np.vstack(failed_tasks)[:, 1])
