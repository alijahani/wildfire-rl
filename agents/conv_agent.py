
import numpy as np, random
from typing import List, Dict, Any

import torch
import torch.nn as nn

from free_range_zoo.utils.agent import Agent
import free_range_rust

import configs, utils

class ConvAgent(Agent):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device = utils.get_device()
        self.actions = torch.zeros((self.parallel_envs, 2), dtype=torch.int32).to(self.device)
        self.step = 0

    def act(self, action_space: free_range_rust.Space) -> List[List[int]]:
        return self.actions

    def observe(self, observation: Dict[str, Any]) -> None:
        self.step += 1
        td_level = configs.args.td_level
        replay_memory = configs.experience_replay[self.agent_name]
        models = configs.models
        criterion = nn.SmoothL1Loss()
        args = configs.args

        self.observation, self.reward, episode, random_indices, loss_list, epsilon, final, configuration, temp_loss = observation
        self.observation, actionmap = self.observation
        actionmap = actionmap['agent_action_mapping']
        action_to_grid = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
        action_to_grid = {k+1: action_to_grid[k] for k in range(len(action_to_grid))}
        grid_to_action = {action_to_grid[k+1]: k+1 for k in range(len(action_to_grid))}

        has_suppressant = self.observation['self'][:, 3] != 0
        tasks = self.observation['tasks']
        self.reward = torch.stack([self.reward[k] for k in self.reward]).permute(1, 0).to(self.device)
        self.sumreward = torch.sum(self.reward, 1)

        current_grid_np = np.zeros((self.parallel_envs, 16, configuration.grid_height, configuration.grid_width))
        only_actions_np = np.zeros((self.parallel_envs, 9))
        action_mask_np = np.ones_like(current_grid_np)
        action_mask_np[:, 3:12, :, :] = 0
        self_np = self.observation['self'].cpu().numpy().astype(int)
        observation_task_np = torch.nested.to_padded_tensor(self.observation['tasks'].cpu(), padding=-1).numpy().astype(int)
        observation_other_np = self.observation['others'].cpu().numpy().astype(int)
        actionmap_np= [[-2]] * self.parallel_envs
        if actionmap.nelement() != 0:
            actionmap_np = torch.nested.to_padded_tensor(actionmap, padding=-2).cpu().numpy().astype(int)           

        for i in range(self.parallel_envs):
            for j, task in enumerate(observation_task_np[i]):
                if task[0] == -1:
                    break
                current_grid_np[i][13][task[0]][task[1]] = 1
                current_grid_np[i][14][task[0]][task[1]] = task[2]
                current_grid_np[i][15][task[0]][task[1]] = task[3]

            for j, other in enumerate(observation_other_np[i]):
                current_grid_np[i][12][other[0]][other[1]] = 1

            current_grid_np[i][0][self_np[i][0]][self_np[i][1]] = 1
            current_grid_np[i][1][self_np[i][0]][self_np[i][1]] = self_np[i][2]
            current_grid_np[i][2][self_np[i][0]][self_np[i][1]] = self_np[i][3]

        possible_list = [{} for i in range(self.parallel_envs)]
        for i in range(self.parallel_envs):
            if actionmap_np[i][0] != -2:
                possible_actions = []
                weakest_fire = 1e10
                weakest_fire_i = -1
                for action in enumerate(actionmap_np[i]):
                    if action[1] == -2:
                        break
                    position = tuple(observation_task_np[i][action[1], 0:2] - self_np[i][0:2])
                    grid_action = grid_to_action[position]
                    possible_actions.append(grid_action)
                    if observation_task_np[i][action[1]][3] < weakest_fire:
                        weakest_fire = observation_task_np[i][action[1]][3]
                        weakest_fire_i = action[0]
                    possible_list[i][grid_action] = action[0]

                # action_index = weakest_fire_i
                action_index = random.randint(0, len(possible_actions) - 1)

                if action_index != len(possible_actions):
                    action = possible_actions[action_index]
                    self.actions[i, 0] = action_index
                    self.actions[i, 1] = 0
                    current_grid_np[i][3+action][self_np[i][0]][self_np[i][1]] = 1
                    only_actions_np[i][action] = 1
                else:
                    self.actions[i, 0] = -1
                    self.actions[i, 1] = -1
                    current_grid_np[i][3][self_np[i][0]][self_np[i][1]]  = 1
                    only_actions_np[i][0] = 1
            else:
                self.actions[i, 0] = -1
                self.actions[i, 1] = -1
                current_grid_np[i][3][self_np[i][0]][self_np[i][1]]  = 1
                only_actions_np[i][0] = 1

        current_grid = torch.from_numpy(current_grid_np).to(self.device)
        only_actions = torch.from_numpy(only_actions_np).to(self.device)
        action_mask = torch.from_numpy(action_mask_np).to(self.device)

        if random.random() > epsilon:
            with torch.no_grad():
                history_length_current = min(args.history_length, self.step - 1)
                if history_length_current > 2:
                    history = replay_memory['history'][:, :, -(history_length_current+1):-1, :, :]
                    e = models.history_encoder(history)
                else:
                    e = torch.zeros((self.parallel_envs, 8, 3, 3)).to(self.device)
                history_and_current = torch.cat((current_grid * action_mask, e), 1).float()
                q = models.q2(history_and_current)
                q_values = torch.clamp(q, min=-1e9, max=1e9)

                for l in range(self.parallel_envs):
                    for i in range(0, 9):
                        if i not in possible_list[l]:
                            q_values[l, i] = -1e10

                q_arg = q_values.argmax(dim=1)

                for l in range(self.parallel_envs):
                    self_d = self.observation['self'][l]
                    action = q_arg[l].item()
                    current_grid[l][3:12][int(self_d[0])][int(self_d[1])] = 0
                    current_grid[l][3+action][int(self_d[0])][int(self_d[1])] = 1
                    only_actions[l][:] = 0
                    only_actions[l][action] = 1
                    if action != 0:
                        self.actions[l, 0] = possible_list[l][action]
                        self.actions[l, 1] = 0
                    else:
                        self.actions[l, 0] = -1
                        self.actions[l, 1] = -1

        for index in random_indices:
            td_possible = True
            if index + 2 + td_level >= args.experience_length or replay_memory['episode'][index] != replay_memory['episode'][index + 1 + td_level]:
                td_possible = False
            if not td_possible or replay_memory['episode'][index] == -1:
                continue

            history_length_index = min(args.history_length, replay_memory['step'][index] - 1, index)

            if history_length_index > 2:
                history = replay_memory['history'][:, :, index-history_length_index:index, :, :]
                e = models.history_encoder(history)
            else:
                e = torch.zeros((self.parallel_envs, 8, 3, 3)).to(self.device)

            history_length_index2 = min(args.history_length, replay_memory['step'][index+td_level] - 1, index+td_level)
            if history_length_index2 > 2:
                history2 = replay_memory['history'][:, :, index+td_level-history_length_index2:index+td_level, :, :]
                ep = models.history_encoder(history2)
            else:
                ep = torch.zeros((self.parallel_envs, 8, 3, 3)).to(self.device)

            state1 = replay_memory['history'][:, :, index, :, :]
            state2 = replay_memory['history'][:, :, index+td_level, :, :]

            g_pred = models.predict_grid(torch.cat((state1, e), 1))
            g_pred[:, 3:12, :, :] = 0
            future_grid = replay_memory['history'][:, :, index+1, :, :].clone()
            future_grid[:, 3:12, :, :] = 0
            g_loss = criterion(g_pred , future_grid)

            state1 = state1.clone()
            state1[:, 3:12, : :] = 0.
            state2 = state2.clone()
            state2[:, 3:12, : :] = 0.

            a = replay_memory['action'][:, index, :]
            ap = replay_memory['action'][:, index+td_level, :]

            q = torch.sum(models.q2(torch.cat((state1, e.detach()), 1)) * a, 1)
            qp = torch.sum(models.q1(torch.cat((state2, ep.detach()), 1)) * ap, 1)
            qp[replay_memory['final'][:, -1].bool()] = 0

            sum_r = q
            for i in range(0, td_level):
                sum_r = sum_r + (args.gamma ** i) * torch.sum(replay_memory['rewards'][:, index+i+1, :], 1)
            expected_q_values = sum_r + (args.gamma ** td_level) * qp
            q_loss = criterion(q, expected_q_values)

            loss_list.append(g_loss)
            loss_list.append(q_loss)
            replay_memory['loss'][index] = q_loss.item()

            temp_loss['g'].append(g_loss.item())
            temp_loss['q'].append(q_loss.item())

        replay_memory['history'] = torch.roll(replay_memory['history'], -1, 2)
        replay_memory['history'][:, :, -1, :, :] = current_grid
        replay_memory['rewards'] = torch.roll(replay_memory['rewards'], -1, 1)
        replay_memory['rewards'][:, -1, :] = self.reward
        replay_memory['final'] = torch.roll(replay_memory['final'], -1, 1)
        replay_memory['final'][:, -1] = final
        replay_memory['action'] = torch.roll(replay_memory['action'], -1, 1)
        replay_memory['action'][:, -1, :] = only_actions
        replay_memory['episode'].pop(0)
        replay_memory['episode'].append(episode)
        replay_memory['step'].pop(0)
        replay_memory['step'].append(self.step)
        replay_memory['loss'].pop(0)
        replay_memory['loss'].append(1e5)
