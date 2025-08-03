import os, pickle, random, logging, numpy as np
from datetime import datetime

import torch
import torch.nn as nn

from free_range_zoo.envs import wildfire_v0
from free_range_zoo.wrappers.action_task import action_mapping_wrapper_v0

import configs, utils
from agents.conv_agent import ConvAgent

def train_loop():
    utils.config_logger()
    args = utils.get_args()
    device = utils.get_device()
    models = configs.models

    r_loss_list = []
    g_loss_list = []
    q_loss_list = []
    rewards_list = []

    epsilon = args.eps_max

    for exec_i in range(1, args.train_episodes+1):
        with open(f'configs/wildfire/WS{(exec_i%3) + 1}.pkl', 'rb') as file:
            configuration = pickle.load(file)
        configuration.fire_config.random_ignition_probability = random.choice([0.05, 0.15, 0.25])
        configuration.fire_config.base_spread_rate = random.choice([0.1, 0.25, 0.5])
        configuration.agent_config.initial_equipment_state = random.choice([0, 2])

        temp_loss = {
            'q': [],
            'g': []
        }
        if exec_i % 1 == 0:
            logging.info(f'{exec_i} th loop start time: {datetime.now().strftime('%H:%M:%S')}')

        env = wildfire_v0.parallel_env(
            parallel_envs=args.parallel_envs,
            max_steps=args.max_steps,
            configuration=configuration,
            device=device,
            show_bad_actions=False,
            observe_other_power=False,
            observe_other_suppressant=False,
        )

        env = action_mapping_wrapper_v0(env)
        observations, infos = env.reset()

        rewards = {
            'firefighter_1': torch.zeros(args.parallel_envs),
            'firefighter_2': torch.zeros(args.parallel_envs),
            'firefighter_3': torch.zeros(args.parallel_envs)
        }

        final = {
            'firefighter_1': torch.zeros(args.parallel_envs),
            'firefighter_2': torch.zeros(args.parallel_envs),
            'firefighter_3': torch.zeros(args.parallel_envs)
        }

        agents = {
            env.agents[0]: ConvAgent(agent_name = 'agent_1', parallel_envs = args.parallel_envs),
            env.agents[1]: ConvAgent(agent_name = 'agent_2', parallel_envs = args.parallel_envs),
            env.agents[2]: ConvAgent(agent_name = 'agent_3', parallel_envs = args.parallel_envs)
        }

        rewards_list_t = []
        while not torch.all(env.finished):
            loss_list = []
            models.optimizer.zero_grad()
            possible_indices = [int(configs.experience_replay['agent_1']['episode'][i] != -1) for i in range(args.experience_length)]
            p = np.array(possible_indices)
            p = p+1e-10
            p = p / sum(p)
            random_indices = np.random.choice(range(args.experience_length), args.sampled_experiences, replace=False, p=p)

            for agent_name, agent in agents.items():
                agent.observe(
                    (observations[agent_name],
                    rewards,
                    exec_i,
                    random_indices,
                    loss_list,
                    epsilon,
                    final[agent_name],
                    configuration,
                    temp_loss
                )
            )

            if len(loss_list) > 0:
                loss = torch.mean(torch.stack(loss_list))
                loss.backward()
                torch.nn.utils.clip_grad_value_(nn.ModuleList(
                    [
                        models.history_encoder,
                        models.predict_grid,
                        models.q2,
                    ]
                ).parameters(), 10)
                models.optimizer.step()

            agent_actions = {
                    agent_name: agents[agent_name].act(action_space = env.action_space(agent_name))
                for agent_name in env.agents
            }

            observations, rewards, terminations, truncations, infos = env.step(agent_actions)
            final = terminations or truncations
            rewards_list_t.append(torch.sum(torch.stack([rewards[k] for k in rewards])).item() / args.parallel_envs)

        q_loss_list.append(np.mean(temp_loss['q']))
        g_loss_list.append(np.mean(temp_loss['g']))
        rewards_list.append(sum(rewards_list_t))

        if exec_i % 1 == 0:
            logging.info(f'g_loss: {np.nanmean(g_loss_list[-3:])}')
            logging.info(f'q_loss: {np.nanmean(q_loss_list[-3:])}')
            logging.info(f'rewards_list: {np.nanmean(rewards_list[-3:])}')

        for t, p in zip(models.q1.parameters(), models.q2.parameters()):
            t.data.copy_(t.data * (1.0 - args.tau) + p.data * args.tau)

        if epsilon >= args.eps_min:
            epsilon *= args.eps_decay
        else:
            epsilon = args.eps_min

        if exec_i % args.save_each_k_episode == 0:
            os.makedirs('saved/models', exist_ok=True)
            torch.save(models.q1.state_dict(), f'saved/models/q1_model_{exec_i}.pt')
            torch.save(models.q2.state_dict(), f'saved/models/q2_model_{exec_i}.pt')
            torch.save(models.history_encoder.state_dict(), f'saved/models/e_model_{exec_i}.pt')
            torch.save(models.predict_grid.state_dict(), f'saved/models/g_model_{exec_i}.pt')
            torch.save(models.predict_reward.state_dict(), f'saved/models/r_model_{exec_i}.pt')
            torch.save(models.optimizer.state_dict(), f'saved/models/o_model_{exec_i}.pt')
        env.close()
