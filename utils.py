import torch
import torch.nn as nn

import gym
import numpy as np
from envs import make_env

# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


# https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py#L87
def init_normc_(weight, gain=1):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))


# Borrowed from openai baselines running_mean_std.py
class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4):
        self.mean = 0.0
        self.var = 1.0 
        self.count = epsilon

    def update(self, x):
        batch_mean = torch.mean(x, axis=0)
        batch_var = torch.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + (delta ** 2) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

def get_env_mean_std(env_id, seed, num_steps=10000):
    import numpy as np
    import torch
    from gym.spaces.box import Box
    from envs import TransposeImage
    from baselines.common.atari_wrappers import make_atari, wrap_deepmind

    if env_id.startswith("dm"):
        _, domain, task = env_id.split('.')
        env = dm_control2gym.make(domain_name=domain, task_name=task)
    else:
        env = gym.make(env_id)
    is_atari = hasattr(gym.envs, 'atari') and isinstance(
        env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
    if is_atari:
        env = make_atari(env_id)
    env.seed(seed)

    obs_shape = env.observation_space.shape

    if is_atari:
        env = wrap_deepmind(env)

    # If the input has shape (W,H,3), wrap for PyTorch convolutions
    obs_shape = env.observation_space.shape
    if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
        env = TransposeImage(env)

    observations = []
    obs = env.reset()
    observations.append(obs)
    for step in range(num_steps):
        act = env.action_space.sample()
        obs, _, done, _ = env.step(act)
        observations.append(obs) 

    observations = np.stack(observations)
    return np.mean(observations), np.std(observations)
