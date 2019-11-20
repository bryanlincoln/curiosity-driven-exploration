import os
import copy
import glob
import time
import types
from collections import deque

import gym
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import algo
import tensorboardX

from envs import make_env, make_vec_envs, VecPyTorch
from utils import get_env_mean_std
from model import Policy, ForwardModel, InverseModel
from storage import RolloutStorage
from arguments import get_args
from visualize import visdom_plot
from tensorboardX import SummaryWriter

import argparse

parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--algo', default='ppo',
                    help='algorithm to use: a2c | ppo | acktr')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='log interval, one log per n updates (default: 10)')
parser.add_argument('--env-name', default='PongNoFrameskip-v4',
                    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument('--load-dir', default='./trained_models/',
                    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument('--log-dir', default='./logs/',
                    help='directory to save agent logs (default: ./logs)')
parser.add_argument('--add-timestep', action='store_true', default=False,
                    help='add timestep to observations')
args = parser.parse_args()

assert args.algo in ['a2c', 'ppo', 'acktr']

torch.manual_seed(args.seed)

eval_log_dir = args.log_dir + "_eval"

try:
    os.makedirs(eval_log_dir)
except OSError:
    pass
#    files = glob.glob(os.path.join(eval_log_dir, '*.monitor.csv'))
#    for f in files:
#        os.remove(f)


def main():
    env = make_vec_envs(args.env_name, args.seed, 1, None,
                        None, args.add_timestep, device='cpu', allow_early_resets=True)

    actor_critic, ob_rms = torch.load(os.path.join(
        args.load_dir, args.algo, args.env_name + ".pt"))

    obs = env.reset()

    eval_episode_rewards = []

    eval_recurrent_hidden_states = torch.zeros(1,
                                               actor_critic.recurrent_hidden_state_size, device='cpu')
    eval_masks = torch.zeros(1, 1, device='cpu')

    while len(eval_episode_rewards) < 10:
        obs = torch.Tensor(obs)
        print('IN1', obs.shape)
        #obs = obs[np.newaxis, :]
        #print('IN2', obs.shape)

        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

        # Obser reward and next obs
        obs, reward, done, infos = env.step(action)
        eval_masks = torch.FloatTensor(
            [[0.0] if done_ else [1.0] for done_ in done])
        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    env.close()

    print(" Evaluation using {} episodes: mean reward {:.5f}\n".
          format(len(eval_episode_rewards),
                 np.mean(eval_episode_rewards)))


if __name__ == "__main__":
    main()
