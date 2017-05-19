# Taken from https://github.com/ikostrikov/pytorch-a3c
from __future__ import absolute_import, division, print_function

import numpy as np

import gym
from gym.spaces.box import Box
from universe import vectorized
from universe.wrappers import Unvectorize, Vectorize

import cv2


# Taken from https://github.com/openai/universe-starter-agent
def create_atari_env(env_id, frame_stack_size=1, noop_init=0, image_dim=42):
    env = gym.make(env_id)
    if len(env.observation_space.shape) > 1:
        print('Preprocessing env')
        env = Vectorize(env)
        env = AtariRescale(env, dim=image_dim)
        env = NormalizedEnv(env)
        if frame_stack_size > 1:
            env = Stack(env, frame_stack_size=frame_stack_size)
        if noop_init > 0:
            env = NoopInit(env, max_n_noops=noop_init)
        env = Unvectorize(env)
    else:
        print('No preprocessing because env is too small')
    return env


def _process_frame(frame, dim=42):
    frame = frame[34:34 + 160, :160]
    # Resize by half, then down to 42x42 (essentially mipmapping). If
    # we resize directly we lose pixels that, when mapped to 42x42,
    # aren't close enough to the pixel boundary.
    if dim < 80:
        frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (dim, dim))
    frame = frame.mean(2)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.reshape(frame, [1, dim, dim])
    return frame


class AtariRescale(vectorized.ObservationWrapper):

    def __init__(self, env=None, dim=42):
        super(AtariRescale, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [1, dim, dim])
        self.dim = dim

    def _observation(self, observation_n):
        return [_process_frame(observation, dim=self.dim) for observation in observation_n]


class NormalizedEnv(vectorized.ObservationWrapper):

    def __init__(self, env=None):
        super(NormalizedEnv, self).__init__(env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.max_episode_length = 0

    def _observation(self, observation_n):
        for observation in observation_n:
            self.max_episode_length += 1
            self.state_mean = self.state_mean * self.alpha + \
                observation.mean() * (1 - self.alpha)
            self.state_std = self.state_std * self.alpha + \
                observation.std() * (1 - self.alpha)

        denom = (1 - pow(self.alpha, self.max_episode_length))
        unbiased_mean = self.state_mean / denom
        unbiased_std = self.state_std / denom

        return [(observation - unbiased_mean) / (unbiased_std + 1e-8)
                for observation in observation_n]


class Stack(vectorized.ObservationWrapper):

    def __init__(self, env=None, frame_stack_size=1):
        super(Stack, self).__init__(env)
        self.frame_stack_size = frame_stack_size
        self.previous_stack = None
    
    def _reset(self):
        self.previous_stack = None
        return super(Stack, self)._reset()

    def _observation(self, observation_n):
        if self.previous_stack is None:
            self.previous_stack = observation_n * self.frame_stack_size
        self.previous_stack = (self.previous_stack + observation_n)[-self.frame_stack_size:]
        return [np.vstack(self.previous_stack)]


class NoopInit(vectorized.ObservationWrapper):

    def __init__(self, env=None, max_n_noops=30):
        super(NoopInit, self).__init__(env)
        self.max_n_noops = max_n_noops
    
    def _reset(self):
        state = super(NoopInit, self)._reset()
        for _ in range(np.random.randint(self.max_n_noops+1)):
            state,_,_,_ = self.step([0])
        return state
    
    def _observation(self, observation_n):
        return observation_n
