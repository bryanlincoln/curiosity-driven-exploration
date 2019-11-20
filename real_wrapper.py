import gym
import math
import numpy as np
import torch
import cv2
from gym.spaces import Box


class RealWrapper(gym.Wrapper):
    def __init__(self, env, time_limit=1000):
        super(RealWrapper, self).__init__(env)
        self.time_limit = time_limit
        self.step_count = 0

        """
        self.observation_space = {
            'linear': Box(-math.pi/2, math.pi/2, (13,)),
            'visual': Box(0, 1, (240, 320, 3))
        }
        """
        self.observation_space = Box(0, 1, (3, 84, 84))

    def reset(self):
        self.step_count = 0
        obs = self.env.reset()
        return self._process_obs(obs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        self.step_count += 1
        if self.step_count > self.time_limit:
            done = True

        return self._process_obs(obs), rew, done, info

    def _process_obs(self, obs):
        del obs['goal']
        """
        _obs = {
            'linear': torch.cat(
                (torch.tensor(obs['joint_positions'], dtype=torch.float16),
                 torch.tensor(obs['touch_sensors'], dtype=torch.float16))
            ),
            'visual': np.array(obs['retina'], dtype=np.float16) / 255
        }
        """
        obs = np.array(obs['retina'], dtype=np.uint8)
        obs = cv2.resize(obs, (84, 84))
        obs = obs.astype(np.float16) / 255
        # obs = np.dot(obs, [0.299, 0.587, 0.114]) # grayscale
        obs = np.moveaxis(obs, 2, 0)
        obs = np.moveaxis(obs, 1, 2)
        return obs
