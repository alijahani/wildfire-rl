import torch

import utils

args = utils.get_args()
device = utils.get_device()

history_zero = torch.zeros((args.parallel_envs, 16, args.experience_length, 3, 3)).to(device)
action_zero = torch.zeros((args.parallel_envs, args.experience_length, 9)).to(device)
rewards_zero = torch.zeros((args.parallel_envs, args.experience_length, 3)).to(device)
final_zero = torch.zeros((args.parallel_envs, args.experience_length)).to(device)

experience_replay = {
    agent: {
        'history': history_zero.clone(),
        'rewards': rewards_zero.clone(),
        'final': final_zero.clone(),
        'action': action_zero.clone(),
        'episode': [-1] * args.experience_length,
        'step': [-1] * args.experience_length,
        'loss': [1e3] * args.experience_length,
    } for agent in ['agent_1', 'agent_2', 'agent_3']
}

models = utils.get_models()
