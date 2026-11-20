#!/usr/bin/env python
# coding: utf-8

import os
import time
import pathlib

project_path = pathlib.Path(__file__).resolve().parent.parent
os.sys.path.insert(1, str(project_path))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from cto.mcts.pvnet import PolicyValueNet, ValueClassifier, pad_collate
from cto.mcts.pvmcts import MCTSDataset

## Load and preprocess data
data = torch.load(f'{project_path}/data/data')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

## CLASSIFIER DATASET --
#
states = []
values = []
action_probs = []
goals = []

for s, v, p, g in data:
    states.append(s)
    action_probs.append(p)
    goals.append(g)
    if v > 0:
        values.append(1)
    else:
        values.append(0)

data_classifier = MCTSDataset(states, values, action_probs, goals)
n_train_classifier = int(0.9 * len(data_classifier))
n_test_classifier = len(data_classifier) - n_train_classifier
train_data_classifier, test_data_classifier = torch.utils.data.random_split(data_classifier,
                                                                            [n_train_classifier,
                                                                             n_test_classifier])
train_loader_classifier = DataLoader(train_data_classifier, batch_size=128,
                                     shuffle=True, collate_fn=pad_collate)

## PVNET DATASET --
#
states = []
values = []
action_probs = []
goals = []

for s, v, p, g in data:
    if v > 0:
        states.append(s)
        values.append(v)
        action_probs.append(p)
        goals.append(g)

data_pvnet = MCTSDataset(states, values, action_probs, goals)
n_train_pvnet = int(0.9 * len(data_pvnet))
n_test_pvnet = len(data_pvnet) - n_train_pvnet
train_data_pvnet, test_data_pvnet = torch.utils.data.random_split(data_pvnet,
                                                                  [n_train_pvnet,
                                                                   n_test_pvnet])
train_loader_pvnet = DataLoader(train_data_pvnet, batch_size=128,
                                shuffle=True, collate_fn=pad_collate)

## TRAINING --
#
# Train classifier --
#
value_classifier = ValueClassifier().to(device)
optimizer = torch.optim.Adam(value_classifier.parameters())
value_classifier.train()

class_ratio = (len(train_data_classifier) - len(train_data_pvnet)) / len(train_data_pvnet)
bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_ratio]).to(device))
tepoch = tqdm(range(300))

for epoch in tepoch:
    running_loss = 0
    n_batches = 0

    for padded_state, state_length, v, p, g in train_loader_classifier:
        optimizer.zero_grad()
        packed_state = pack_padded_sequence(padded_state, state_length,
                                            enforce_sorted=False, batch_first=True)
        y_pred = value_classifier(packed_state.to(device).float(),
                                  g.to(device).float())
        loss = bce_loss(y_pred, v.to(device).float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        n_batches += 1

    tepoch.set_postfix(epoch_loss=running_loss / n_batches)

# Train pvnet --
#
pvnet = PolicyValueNet().to(device)
optimizer = torch.optim.Adam(pvnet.parameters())
pvnet.train()
tepoch = tqdm(range(300))

for epoch in tepoch:
    running_loss = 0
    n_batches = 0
    for padded_state, state_length, v, p, g in train_loader_pvnet:
        optimizer.zero_grad()
        packed_state = pack_padded_sequence(padded_state, state_length,
                                            enforce_sorted=False, batch_first=True)
        pi_pred, v_pred = pvnet(packed_state.to(device).float(), g.to(device).float())
        value_loss = F.mse_loss(v.to(device).float(), v_pred)
        policy_loss = -torch.mean(torch.sum(p.to(device) * pi_pred, 1))
        loss = value_loss + policy_loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        n_batches += 1
    tepoch.set_postfix(epoch_loss=running_loss / n_batches)

## Testing
value_classifier.eval()
Y = []
Y_pred = []

for s, v, p, g in test_data_classifier:
    s = torch.tensor(s)[None, :, :].to(device).float()
    g = torch.tensor(g)[None, :].to(device).float()
    y_pred = torch.sigmoid(value_classifier(s, g))
    y_pred = np.squeeze((y_pred.detach().cpu().numpy()))
    Y.append(v)
    Y_pred.append(y_pred)
Y_pred = np.array(Y_pred)
Y = np.array(Y)
class0 = Y_pred[Y == 0]
class1 = Y_pred[Y == 1]

# compute confusion to fine tune the threshold if needed
# high TN -> prune infeasible traj. more aggresively
th = 0.5
print('True negative:', len(class0[class0 <= th]) / len(class0))
print('False negative:', len(class1[class1 <= th]) / len(class1))
print('True positive:', len(class1[class1 >= th]) / len(class1))
print('False positive:', len(class0[class0 >= th]) / len(class0))

# visualize the regression results
pvnet.eval()
V = []
V_pred = []
for s, v, p, g in test_data_pvnet:
    s = torch.tensor(s)[None, :, :].to(device).float()
    g = torch.tensor(g)[None, :].to(device).float()
    pi_pred, v_pred = pvnet(s, g)
    v_pred = np.squeeze((v_pred.detach().cpu().numpy()))
    V.append(v)
    V_pred.append(v_pred)

plt.scatter(V, V_pred)

## Save the trained models
torch.save(value_classifier.state_dict(), f'{project_path}/models/value_classifier')
torch.save(pvnet.state_dict(), f'{project_path}/models/pvnet')
