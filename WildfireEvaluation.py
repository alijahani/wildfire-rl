"""Evaluate wildfire baselines on testing configurations."""

import warnings

warnings.simplefilter('ignore', UserWarning)

import argparse
import os
import sys
import torch
import logging
import pickle

from free_range_zoo.envs import wildfire_v0
from free_range_zoo.wrappers.action_task import action_mapping_wrapper_v0
from free_range_zoo.envs.wildfire.baselines import NoopBaseline, RandomBaseline, StrongestBaseline, WeakestBaseline

FORMAT_STRING = "[%(asctime)s] [%(levelname)8s] [%(name)10s] [%(filename)21s:%(lineno)03d] %(message)s"

# ADDITION

from typing import List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import free_range_rust
from free_range_zoo.utils.agent import Agent
import random
import torch.optim as optim
from torch.distributions.uniform import Uniform
import time
import random, math

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.autograd import Variable

import numpy as np

history_length = 30
experience_length = 35

class SEResidual3D(nn.Module):
    def __init__(self, channel_num, reduction = 4):
        super(SEResidual3D, self).__init__()
        self.conv1 = nn.Conv3d(
            channel_num, channel_num, 3, padding='same'
        )
        self.norm1 = nn.BatchNorm3d(channel_num)
        self.conv2 = nn.Conv3d(
            channel_num, channel_num, 3, padding='same'
        )
        self.norm2 = nn.BatchNorm3d(channel_num)
        self.fc1 = nn.Linear(channel_num, channel_num // reduction)
        self.fc2 = nn.Linear(channel_num // reduction, channel_num)

    def forward(self, x):
        residual = x
        x = F.silu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))

        scale = F.avg_pool3d(x, kernel_size=x.size()[2:5])[:, :, 0, 0, 0]
        scale = F.silu(self.fc1(scale))
        scale = F.sigmoid(self.fc2(scale))
        scale = scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = x * scale
        return F.silu(residual + x)

class SEResidual2D(nn.Module):
    def __init__(self, channel_num, reduction = 4):
        super(SEResidual2D, self).__init__()
        self.conv1 = nn.Conv2d(
            channel_num, channel_num, 3, padding='same'
        )
        self.norm1 = nn.BatchNorm2d(channel_num)
        self.conv2 = nn.Conv2d(
            channel_num, channel_num, 3, padding='same'
        )
        self.norm2 = nn.BatchNorm2d(channel_num)

        self.fc1 = nn.Linear(channel_num, channel_num // reduction)
        self.fc2 = nn.Linear(channel_num // reduction, channel_num)

    def forward(self, x):
        residual = x

        x = F.silu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))

        scale = F.avg_pool2d(x, kernel_size=x.size()[2:4])[:, :, 0, 0]
        scale = F.silu(self.fc1(scale))
        scale = F.sigmoid(self.fc2(scale))
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        x = x * scale

        return F.silu(residual + x)

class HistoryEncoder(nn.Module):
    def __init__(self):
        super(HistoryEncoder, self).__init__()
        self.conv1 = nn.Conv3d(
            16, 16, 3, padding='same'
        )
        self.block1 = SEResidual3D(16)
        self.block2 = SEResidual3D(16)
        self.conv2 = nn.Conv3d(
            16, 8, 3, padding='same'
        )
        self.conv3 = nn.Conv3d(
            16, 8, 3, padding='same'
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)

        a1 = F.silu(torch.mean(self.conv2(x), 2))
        a2 = F.silu(self.conv3(x)[:, :, -1, :, :])

        return a1 + a2

class PredictGrid(nn.Module):
    def __init__(self):
        super(PredictGrid, self).__init__()
        self.conv1 = nn.Conv2d(
            24, 64, 3, padding='same'
        )
        self.block1 = SEResidual2D(64)
        self.block2 = SEResidual2D(64)
        self.conv2 = nn.Conv2d(
            64, 16, 3, padding='same'
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.conv2(x)
        x[:, 13, :, :] = F.sigmoid(x[:, 13, :, :])
        return x

class PredictReward(nn.Module):
    def __init__(self):
        super(PredictReward, self).__init__()
        self.conv1 = nn.Conv2d(
            24, 64, 3, padding='same'
        )
        self.block1 = SEResidual2D(64)
        self.block2 = SEResidual2D(64)
        self.conv2 = nn.Conv2d(
            64, 3, 1, padding='same'
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.conv2(x)
        x = torch.sum(x, (2,3))
        return x

class PredictQ(nn.Module):
    def __init__(self):
        super(PredictQ, self).__init__()
        self.conv1 = nn.Conv2d(
            24, 64, 3, padding='same'
        )
        self.block1 = SEResidual2D(64)

        self.fc1 = nn.Linear(64, 9)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)

        x = F.silu(torch.sum(x, (2,3)))
        a = self.fc1(x)
        a = a - torch.mean(a, 1, keepdim = True)
        v = self.fc2(x)
        return a + v


class GuidedSearchActor(Agent):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.actions = torch.zeros((self.parallel_envs, 2), dtype=torch.int32).to(device)
        self.step = 0

    def act(self, action_space: free_range_rust.Space) -> List[List[int]]:
        return self.actions

    def observe(self, observation: Dict[str, Any]) -> None:
        self.step += 1
        replay_memory = experience_replay[self.agent_name]

        self.observation = observation
        self.observation, actionmap = self.observation
        actionmap = actionmap['agent_action_mapping']
        action_to_grid = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
        action_to_grid = {k+1: action_to_grid[k] for k in range(len(action_to_grid))}
        grid_to_action = {action_to_grid[k+1]: k+1 for k in range(len(action_to_grid))}

        has_suppressant = self.observation['self'][:, 3] != 0
        tasks = self.observation['tasks']

        current_grid_np = np.zeros((self.parallel_envs, 16, 3, 3))
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

        current_grid = torch.from_numpy(current_grid_np).to(device)
        only_actions = torch.from_numpy(only_actions_np).to(device)
        action_mask = torch.from_numpy(action_mask_np).to(device)

        if True:
            with torch.no_grad():
                history_length_current = min(history_length, self.step - 1)
                if history_length_current > 2:
                    history = replay_memory['history'][:, :, -(history_length_current+1):-1, :, :]
                    e = e_model(history)
                else:
                    e = torch.zeros((self.parallel_envs, 8, 3, 3)).to(device)
                history_and_current = torch.cat((current_grid * action_mask, e), 1).float().to(device)
                q = q2_model(history_and_current)
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

        replay_memory['history'] = torch.roll(replay_memory['history'], -1, 2)
        replay_memory['history'][:, :, -1, :, :] = current_grid

# END OF ADDITION

def main() -> None:
    """Run the training experiment."""
    global device, args, dataset
    args = handle_args()
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    ### ADDED CODE: LOADING WEIGHTS
    global e_model, g_model, r_model, q1_model, q2_model, experience_replay
    e_model = HistoryEncoder().to(device) # Encoder model
    g_model = PredictGrid().to(device) # Grid model
    r_model = PredictReward().to(device) # Reward model
    q1_model = PredictQ().to(device) # state-action model
    q2_model = PredictQ().to(device) # state-action model
    q2_model.load_state_dict(q1_model.state_dict())

    n = args.model_to_load
    e_model.load_state_dict(torch.load(f'saved/models/e_model_{n}.pt', weights_only=True))
    g_model.load_state_dict(torch.load(f'saved/models/g_model_{n}.pt', weights_only=True))
    r_model.load_state_dict(torch.load(f'saved/models/r_model_{n}.pt', weights_only=True))
    q1_model.load_state_dict(torch.load(f'saved/models/q1_model_{n}.pt', weights_only=True))
    q2_model.load_state_dict(torch.load(f'saved/models/q2_model_{n}.pt', weights_only=True))

    for param in q1_model.parameters():
        param.requires_grad = False

    bceloss = nn.BCELoss()
    celoss = nn.CrossEntropyLoss()
    criterion = nn.SmoothL1Loss()

    models_list = [e_model, r_model, g_model, q2_model]
    parallel_envs = args.testing_episodes

    history_zero = torch.zeros((parallel_envs, 16, experience_length, 3, 3)).to(device)
    action_zero = torch.zeros((parallel_envs, experience_length, 9)).to(device)
    rewards_zero = torch.zeros((parallel_envs, experience_length, 3)).to(device)

    experience_replay = {
        agent: {
            'history': history_zero.clone(),
            'rewards': rewards_zero.clone(),
            'action': action_zero.clone(),
            'episode': [-1] * experience_length,
            'step': [-1] * experience_length,
            'loss': [1e3] * experience_length
        } for agent in ['firefighter_1', 'firefighter_2', 'firefighter_3']
    }
    ### END OF LOADING WEIGHTS

    if args.threads > 1:
        torch.set_num_threads(args.threads)

    os.makedirs(args.output, exist_ok=True)

    if os.path.exists(args.output):
        main_logger.warning(f'Output directory {args.output} already exists and may contain artifacts from a previous run.')

    main_logger.info(f'Running the baseline experiment on device {device} and parameters:')
    for key, value in vars(args).items():
        main_logger.info(f'- {key}: {value}')

    torch.use_deterministic_algorithms(True, warn_only=True)
    generator = torch.Generator()
    generator = generator.manual_seed(args.seed)
    torch.manual_seed(torch.randint(0, 100000000, (1, ), generator=generator).item())

    main_logger.info('Initializing dataset')

    try:
        test()
    except KeyboardInterrupt:
        main_logger.warning('Testing interrupted by user')
    except Exception as e:
        main_logger.error(f'Error during testing: {e}')
        raise e


@torch.no_grad()
def test() -> None:
    """
    Run the testing episodes for the model.

    Args:
        model: nn.Module - The model to validate.
    """
    # TODO: For other domains, swap environment initialization and agent names
    agents = ['firefighter_1', 'firefighter_2', 'firefighter_3']
    with open(args.config, 'rb') as f: # CHANGED: pickle.load(f), f should be a file.
        env = wildfire_v0.parallel_env(
          parallel_envs=args.testing_episodes,
        	max_steps=50,
        	device=device,
        	configuration=pickle.load(f),
        	buffer_size=50,
        	single_seeding=True,
        	show_bad_actions=False,
	    )

    env = action_mapping_wrapper_v0(env)
    observation, _ = env.reset(seed=0)

    agents = {}
    for agent_name in env.agents:
        agents[agent_name] = GuidedSearchActor(agent_name = agent_name, parallel_envs = args.testing_episodes) # TODO: Call policy class initialization to initialize each agent

    step = 0
    global_total_rewards = {agent: torch.zeros(args.testing_episodes).to(device) for agent in agents} #CHANGED ADDED it was missing
    total_rewards = {agent: torch.zeros(args.testing_episodes).to(device) for agent in agents} #CHANGED "to device"

    while not torch.all(env.finished):
        test_logger.info(f'STEP {step}')
        agent_actions = {}
        for agent_name, agent_model in agents.items():
            agent_model.observe(observation[agent_name])

            actions = agent_model.act(env.action_space(agent_name))
            actions = torch.tensor(actions, device=device, dtype=torch.int32)
            agent_actions[agent_name] = actions

        observation, reward, term, trunc, info = env.step(agent_actions)

        test_logger.info('ACTIONS')
        for batch in range(args.testing_episodes):
            batch_actions = ' '.join(f'{agent_name}: {str(agent_actions[batch].tolist()):<10}\t'
                                     for agent_name, agent_actions in agent_actions.items())
            test_logger.info(f'{batch + 1}:\t{batch_actions}')

        test_logger.info('REWARDS')
        for agent_name in env.agents:
            test_logger.info(f'{agent_name}: {reward[agent_name]}')
            total_rewards[agent_name] += reward[agent_name]

        step += 1

    for agent_name, total_reward in total_rewards.items():
        global_total_rewards[agent_name] += total_reward

    test_logger.info('TOTAL REWARDS') #CHANGED
    for agent_name, total_reward in total_rewards.items():
        test_logger.info(f'{agent_name}: {total_reward}')

    totals = torch.zeros(args.testing_episodes).to(device)
    for agent, reward_tensor in total_rewards.items():
        totals += reward_tensor

    total_mean = round(totals.mean().item(), 3)
    total_std_dev = round(totals.std().item(), 3)

    average_mean = round(total_mean / len(agents), 3)
    average_std_dev = round(total_std_dev / len(agents), 3)

    test_logger.info('REWARD SUMMARY') #CHANGED
    test_logger.info(f'Average Reward: {average_mean} ± {average_std_dev}')
    test_logger.info(f'Total Reward: {total_mean} ± {total_std_dev}')

def handle_args() -> argparse.Namespace:
    """
    Handle script arguments.

    Returns:
        argparse.Namespace - parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description='Run baseline policies on a given wildfire configuration.')

    general = parser.add_argument_group('General')
    general.add_argument('output', type=str, help='output directory for all experiments artifacts')
    general.add_argument('config', type=str, help='path to environment configuration to utilize')
    general.add_argument('--model-to-load', type=int, help='Which model to load for evaluation')
    general.add_argument('--cuda', action='store_true', help='Utilize cuda if available')
    general.add_argument('--threads', type=int, default=1, help='utilize this many threads for the experiment')

    reproducible = parser.add_argument_group('Reproducibility')
    reproducible.add_argument('--seed', type=int, default=None, help='seed for the experiment')
    reproducible.add_argument('--dataset_seed', type=int, default=None, help='seed for initializing the configuration dataset')

    validation = parser.add_argument_group('Validation')
    validation.add_argument('--testing_episodes', type=int, default=16, help='number of episodes to run per test') 

    return parser.parse_args()

os.makedirs('saved/logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s (%(levelname)s): %(message)s",
    handlers=[
        logging.FileHandler("saved/logs/eval.log"),
        logging.StreamHandler()
    ]
)

main_logger = logging.getLogger('main') #TODO
test_logger = logging.getLogger('baseline')

if __name__ == '__main__':
    main()
    sys.exit()
