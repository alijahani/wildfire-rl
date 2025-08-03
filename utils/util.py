import logging, argparse, os
from collections import namedtuple

import torch.optim as optim
import torch.nn as nn
import torch

from models import encoders, predictors

def get_args():
    parser = argparse.ArgumentParser(prog='wildfire-rl')
    parser.add_argument('--history-length', type=int, default=30) # Length to consider when encoding history
    parser.add_argument('--experience-length', type=int, default=1000) # Length to consider for experience relay
    parser.add_argument('--parallel-envs', type=int, default=10)
    parser.add_argument('--sampled-experiences', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.99) # For td learning
    parser.add_argument('--lr', type=float, default=5e-3) # learning rate
    parser.add_argument('--tau', type=float, default=1e-1) # Soft update of target network
    parser.add_argument('--td-level', type=int, default=2)
    parser.add_argument('--eps-max', type=float, default=1.0)
    parser.add_argument('--eps-decay', type=float, default=0.99)
    parser.add_argument('--eps-min', type=float, default=0.01)
    parser.add_argument('--cuda', type=bool, default=True)
    # parser.add_argument('--eval', type=bool, default=False) # TODO
    parser.add_argument('--train-episodes', type=int, default=1000)
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--save-each-k-episode', type=int, default=10)
    parser.add_argument('--load-model', type=int, default=None)

    return parser.parse_args()

def get_device():
    args = get_args()
    return torch.device('cuda' if (torch.cuda.is_available() and args.cuda) else 'cpu')

def get_models():
    device = get_device()
    args = get_args()

    Models = namedtuple('Models', [
        'history_encoder',
        'predict_grid',
        'predict_reward',
        'q1',
        'q2',
        'optimizer'
    ])

    e_model = encoders.HistoryEncoder().to(device) # Encoder model
    g_model = predictors.PredictGrid().to(device) # Grid model
    r_model = predictors.PredictReward().to(device) # Reward model
    q1_model = predictors.PredictQ().to(device) # state-action model
    q2_model = predictors.PredictQ().to(device) # state-action model
    q1_model.load_state_dict(q2_model.state_dict())
    for param in q1_model.parameters():
        param.requires_grad = False

    models_list = [e_model, r_model, g_model, q2_model]
    optimizer = optim.Adam(nn.ModuleList(models_list).parameters(), lr=args.lr)

    if args.load_model is not None:
        e_model.load_state_dict(torch.load(f'saved/models/e_model_{args.load_model}.pt', weights_only=True))
        g_model.load_state_dict(torch.load(f'saved/models/g_model_{args.load_model}.pt', weights_only=True))
        r_model.load_state_dict(torch.load(f'saved/models/r_model_{args.load_model}.pt', weights_only=True))
        q1_model.load_state_dict(torch.load(f'saved/models/q1_model_{args.load_model}.pt', weights_only=True))
        q2_model.load_state_dict(torch.load(f'saved/models/q2_model_{args.load_model}.pt', weights_only=True))
        optimizer.load_state_dict(torch.load(f'saved/models/o_model_{args.load_model}.pt', weights_only=True))

    return Models(e_model, g_model, r_model, q1_model, q2_model, optimizer)

def config_logger():
    os.makedirs('saved/logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s (%(levelname)s): %(message)s",
        handlers=[
            logging.FileHandler("saved/logs/train.log"),
            logging.StreamHandler()
        ]
    )
