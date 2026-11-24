from __future__ import annotations
from typing import Optional
import numpy as np
import os
import logging

# pip install nx-cugraph-cu13 --extra-index-url https://pypi.nvidia.com
os.environ['NX_CUGRAPH_AUTOCONFIG'] = '1'
os.environ['NETWORKX_BACKEND_PRIORITY'] = 'cugraph, networkx'

import networkx as nx

# Enable debug logging for NetworkX dispatching
"""
nxl = logging.getLogger("networkx")
nxl.addHandler(logging.StreamHandler())
nxl.setLevel(logging.DEBUG)
"""

print('NETWORKX BACKEND', nx.config.backends, nx.betweenness_centrality.backends)
import torch
import pinocchio as pin
from torch.utils.data import Dataset
from scipy.special import softmax

from dotmap import DotMap

from cto.mcts.problems import BiconvexProblem, integrate_solution
from cto.heuristics import kinematic_feasibility, contact_removed, no_contact, full_contact
from cto.contact_modes import *
from cto.mcts.pvnet import PolicyValueNet, ValueClassifier

eps = 1e-6


class PolicyValuePredictor:
    def __init__(self, num_actions: int, num_contacts: int, classifier_threshold: float,
                 networks: Optional[list["nn.Module"]] = None):
        self.num_actions = num_actions
        self.num_contacts = num_contacts
        self.classifier_threshold = classifier_threshold

        # create networks if none given
        self.pvnet: PolicyValueNet = None
        self.value_classifier: ValueClassifier = None
        if networks is None:
            self.create_networks()
        else:
            self.load_networks(*networks)

    def load_networks(self, pvnet: PolicyValueNet, value_classifier: ValueClassifier):
        self.pvnet = pvnet
        self.value_classifier = value_classifier

    def create_networks(self, device='cpu'):
        input_size = self.num_contacts
        action_size = self.num_actions
        self.pvnet = PolicyValueNet(input_size, action_size).to(device)
        self.value_classifier = ValueClassifier(input_size, action_size).to(device)

    def load_pvnet(self, model_path, device='cpu'):
        self.pvnet.load_state_dict(torch.load(model_path, map_location=device))

    def load_value_classifier(self, model_path, device='cpu'):
        self.value_classifier.load_state_dict(torch.load(model_path, map_location=device))

    def predict(self, goal: np.ndarray, state: list[list[int]]):
        self.pvnet.eval()
        self.value_classifier.eval()
        with torch.no_grad():
            state = torch.tensor(state).float()[None, :, :]
            goal = torch.tensor(goal).float()[None, :]
            y = self.value_classifier(state, goal)

            if torch.sigmoid(y) > self.classifier_threshold:  # tune this threshold on the test set
                p_pred, v_pred = self.pvnet(state, goal)
                p_pred = np.exp(p_pred.detach().cpu().numpy().squeeze())
                v_pred = v_pred.detach().cpu().numpy().item()
            else:
                p_pred = np.ones(self.num_actions) / self.num_actions
                v_pred = 0
        return p_pred, v_pred


class PolicyValueMCTS(object):

    def __init__(self, params: DotMap, env,
                 uct_coeff=0.5, softmax_coeff=0.1,
                 networks: Optional[list["nn.Module"]] = None):
        self.params = params
        self.env = env
        self.search_tree = nx.DiGraph()
        self.uct_coeff = uct_coeff
        self.softmax_coeff = softmax_coeff
        self.infeasible_contact_plans = set([])

        self.action_ids = {}
        all_contact_modes = enumerate_contact_modes(params)
        for i, mode in enumerate(all_contact_modes):
            self.action_ids[tuple(mode)] = i
        self.num_actions = len(self.action_ids)
        self.legal_actions = {}

        self.reached_nodes_reward = {}
        self.reached_nodes_error = {}
        self.reached_nodes_sol = {}
        self.reached_nodes_state = {}

        # solve for a dummy state to warmup casadi compilation
        dummy_state = [[0] * params.n_contacts] * params.n_contact_modes
        contact_plan = construct_contact_plan(dummy_state, self.params)
        self.biconvex_problem = BiconvexProblem(contact_plan, self.params)
        self.biconvex_problem.setup()
        try:
            _ = self.biconvex_problem.solve()
        except:
            pass

        self.budget = 100
        self.verbose = False

        # trainer
        self.training = False
        self.predictor = PolicyValuePredictor(self.num_actions, self.params.n_contacts,
                                              self.params.classifier_threshold, networks)

    def is_initial(self, state: list[list[int]]):
        if len(state) == 1:
            return True
        else:
            return False

    def is_terminal(self, state: list[list[int]]):
        if len(state) == self.params.n_contact_modes:
            return True
        else:
            return False

    def is_dynamic_mode(self, mode_idx):
        """Mode `mode_idx` belongs to the dynamic phase of its segment (each segment starts with
        n_modes_static static modes followed by n_modes_dynamic dynamic ones)."""
        return (mode_idx % self.params.n_modes_segment) >= self.params.n_modes_static

    def legal_actions_at(self, state: list[list[int]]):
        state_key = np.asarray(state).tobytes()
        if state_key in self.legal_actions.keys():
            return self.legal_actions[state_key]

        if len(state) > 1:
            curr_mode = state[-1]
        else:
            curr_mode = state[0]

        legal_actions = []
        if self.is_terminal(state):
            return legal_actions

        d = self.params.contact_duration
        feasible_transitions = find_feasible_transitions(curr_mode, self.params)
        # optional: during the dynamic phase of a segment, require every manipulator to be in contact
        # (otherwise a one-arm push that already reduces the pose error is accepted as a solution)
        require_full = self.params.require_full_contact
        for next_mode in feasible_transitions:
            n = len(state) * d
            if require_full and self.is_dynamic_mode(len(state)) and not full_contact(next_mode):
                continue
            if kinematic_feasibility(next_mode, n, d, self.params, self.env):
                if contact_removed(curr_mode, next_mode):
                    # only allow removing contact for zero acc.
                    acc_norm = np.linalg.norm(self.params.traj_desired.ddq[n:n + d, :])
                    if np.isclose(acc_norm, 0):
                        legal_actions.append(next_mode)
                else:
                    legal_actions.append(next_mode)

        self.legal_actions[state_key] = legal_actions
        return legal_actions

    def apply_action_to(self, state, action):
        next_state = [*state, action]
        return next_state

    def add_node(self, state: list[list[int]]):
        legal_actions = self.legal_actions_at(state)
        p, v = self.predictor.predict(self.get_state_goal(state), state)

        Q = {tuple(action): 0 for action in legal_actions}
        N = {tuple(action): 0 for action in legal_actions}
        P = {tuple(action): p[self.action_ids[tuple(action)]] for action in legal_actions}

        self.search_tree.add_node(np.asarray(state).tobytes(), state=np.asarray(state), Q=Q, N=N, P=P)
        return v

    def add_edge(self, state, next_state, action):
        self.search_tree.add_edge(np.asarray(state).tobytes(), np.asarray(next_state).tobytes(), action=action)

    def train(self, state, budget=100, max_it=100000, verbose=False):
        # this does not train the NNs
        self.training = True
        self.budget = budget
        self.verbose = verbose

        for i in range(max_it):
            if self.budget <= 0:
                if self.verbose:
                    print('# found solutions:', len(self.reached_nodes_reward))
                return
            else:
                v = self.simulate(state)

    def run(self, state, budget=100, n_candidates=1, max_it=100000, verbose=False):
        self.training = False
        self.verbose = verbose
        self.budget = budget

        for i in range(max_it):
            if self.budget <= 0:
                return
            if len(self.reached_nodes_reward) >= n_candidates:
                return
            v = self.simulate(state)

    def get_solution(self):
        if len(self.reached_nodes_reward) > 0:
            best_node = max(self.reached_nodes_reward, key=self.reached_nodes_reward.get)
            sol = self.reached_nodes_sol[best_node]
            state = self.reached_nodes_state[best_node]
            print('Solved', state)
            return state, sol
        else:
            return None, None

    def act(self, state: list[list[int]]) -> list[int]:
        legal_actions = self.legal_actions_at(state)
        if len(legal_actions) == 0:
            return None

        U = np.array([self.uct(state, action) for action in legal_actions])
        if self.training:
            # softmax
            prob = softmax(U / self.softmax_coeff)
            chosen_idx = np.random.choice(len(legal_actions), p=prob)
        else:
            # max with random tie breaking
            chosen_idx = np.random.choice(np.where(U == U.max())[0])

        action = legal_actions[chosen_idx]

        return action

    def simulate(self, state: list[list[int]]):

        if self.is_terminal(state):
            return self.compute_reward(state)

        if np.asarray(state).tobytes() not in self.search_tree.nodes():
            print("HEURISTIC Simulation step")
            v1 = self.add_node(state)
            v2 = self.heuristic_simulate(state)
            return np.max((v1, v2))

        else:
            print("Simulation step")
            action = self.act(state)
            if action is None:
                return 0

            next_state = self.apply_action_to(state, action)
            v = self.simulate(next_state)
            N, Q, _ = self.get_state_nodes(state)

            Qsa = Q[tuple(action)]
            Nsa = N[tuple(action)]

            Q[tuple(action)] = (Nsa * Qsa + v) / (Nsa + 1)
            N[tuple(action)] += 1

            return v

    def heuristic_simulate(self, state: list[list[int]]):
        while True:
            if self.is_terminal(state):
                reward = self.compute_reward(state)
                break

            else:
                legal_actions = self.legal_actions_at(state)
                if len(legal_actions) != 0:
                    next_states = [self.apply_action_to(state, action) for action in legal_actions]
                    heuristics_score = np.array([self.compute_heuristics_score(state) for state in next_states])

                    prob = softmax(heuristics_score / self.softmax_coeff)
                    random_idx = np.random.choice(len(next_states), p=prob)
                    state = next_states[random_idx]
                else:
                    return 0

        return reward

    def compute_heuristics_score(self, state: list[list[int]]):
        _, v = self.predictor.predict(self.get_state_goal(state), state)
        return v

    def get_state_nodes(self, state: list[list[int]]):
        state_key = np.asarray(state).tobytes()
        node = self.search_tree.nodes[state_key]
        return node['N'], node['Q'], node['P']

    def get_state_goal(self, state: list[list[int]]):
        d = self.params.contact_duration
        future_horizon = self.params.n_modes_segment * d
        idx_prev = (len(state) - 1) * d
        idx_future = np.min((idx_prev + future_horizon - 1, self.params.horizon - 1))
        pose_prev = pin.XYZQUATToSE3(self.params.traj_desired.q[idx_prev, :])
        pose_future = pin.XYZQUATToSE3(self.params.traj_desired.q[idx_future, :])
        pose_diff = pin.log6(pose_prev.actInv(pose_future)).vector
        goal = np.hstack((pose_prev.translation,
                          pin.log3(pose_prev.rotation),
                          pose_diff))
        return goal

    def get_data(self):
        states = []
        values = []
        action_probs = []
        goals = []

        for state_key in self.search_tree.nodes():
            state = self.search_tree.nodes[state_key]['state']
            N, Q, P = self.get_state_nodes(state)
            goal = self.get_state_goal(state)

            action_prob = np.zeros(len(self.action_ids))
            action_value = np.zeros(len(self.action_ids))

            for action, na in N.items():
                action_id = self.action_ids[action]
                action_prob[action_id] = na
                action_value[action_id] = Q[action]

            action_prob = action_prob / (np.sum(action_prob) + eps)
            action_value = action_prob * action_value  # weight action value by its frequency
            value = sum(action_value)

            action_probs.append(action_prob)
            states.append(state)
            values.append(value)
            goals.append(goal)

        return states, values, action_probs, goals

    def uct(self, state, action, eps=1e-6):

        N, Q, P = self.get_state_nodes(state)
        Nsa = N[tuple(action)]
        Nsum = sum(N.values())
        exploitation_score = Q[tuple(action)]
        exploration_score = self.uct_coeff * P[tuple(action)] * np.sqrt(1 + Nsum) / (1 + Nsa)
        U = exploitation_score + exploration_score

        return U

    def compute_reward(self, state: list[list[int]]):

        state_key = np.asarray(state).tobytes()
        # assert self.reached_nodes_reward
        if state_key in self.reached_nodes_reward.keys():
            reward = self.reached_nodes_reward[state_key]
            if self.verbose:
                print('reached a previously found solution with reward:', reward)

        elif state_key in self.infeasible_contact_plans:
            reward = 0

        else:
            self.budget -= 1
            contact_plan = construct_contact_plan(state, self.params)
            try:
                self.biconvex_problem.update(contact_plan)
                sol = self.biconvex_problem.solve(verbose=True)
                reward = self.evaluate_solution(state, sol)
                if self.verbose:
                    print("try contact plan of state:", state, "budget left:", self.budget, 'reward:', reward)
            except Exception as e:
                print(e)
                # this contact plan does not produce feasible solution
                self.infeasible_contact_plans.add(state_key)
                reward = 0

        return reward

    def evaluate_solution(self, state, sol):
        state_key = np.asarray(state).tobytes()
        err_threshold = self.params.err_threshold

        # integrate solution
        reward = 0
        n_segments = self.params.n_desired_poses - 1
        all_pos_err = []
        all_orn_err = []

        print('n_segments', n_segments)
        for i in range(n_segments):
            idx_start = i * (self.params.contact_duration * self.params.n_modes_segment)
            idx_end = (i + 1) * (self.params.contact_duration * self.params.n_modes_segment)

            traj_actual = integrate_solution(sol, self.params, idx_start, idx_end)
            pose_end = pin.XYZQUATToSE3(traj_actual.q[-1])
            pose_des = pin.XYZQUATToSE3(self.params.traj_desired.q[idx_end - 1])

            pos_err = np.linalg.norm(pose_end.translation - pose_des.translation)
            orn_err = np.linalg.norm(pin.log3(pose_end.rotation.T @ pose_des.rotation))

            err = pos_err + 0.1 * orn_err

            all_pos_err.append(pos_err)
            all_orn_err.append(orn_err)

            if err <= err_threshold:
                reward += (np.exp(err_threshold - err) - 1) / (np.exp(err_threshold) - 1)  # normalized to 0-1
                print('REWARD', reward)
            else:
                # if any segment is infeasible the plan is infeasible
                self.infeasible_contact_plans.add(state_key)
                return 0

        reward = reward / n_segments

        if reward > 0:
            self.reached_nodes_sol[state_key] = sol
            self.reached_nodes_state[state_key] = state
            self.reached_nodes_reward[state_key] = reward
            self.reached_nodes_error[state_key] = (100 * np.sum(all_pos_err), 180 / np.pi * np.sum(all_orn_err))
            if self.verbose:
                print('found a solution with reward:', reward, 'State:', state)
        return reward


class MCTSDataset(Dataset):
    def __init__(self, states, values, action_probs, goals):
        super().__init__()
        self.states = states
        self.values = values
        self.action_probs = action_probs
        self.goals = goals

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        return self.states[idx], self.values[idx], self.action_probs[idx], self.goals[idx]
