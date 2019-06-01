'''
GymMDPClass.py: Contains implementation for MDPs of the Gym Environments.
'''

# Python imports.
import random
import sys
import os
import copy
import numpy as np
from scipy.misc import imresize
# import cv2 as cv

# Other imports.
import gym
# import simple_rl.tasks.gym.mujoco
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.gym.GymStateClass import GymState
from collections import defaultdict

class GymMDP(MDP):
    ''' Class for Gym MDPs '''

    def __init__(self, env_name='CartPole-v0', grayscale=False, downscale=False, flatten=True, render=False):
        '''
        Args:
            env_name (str)
        '''
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.grayscale = grayscale
        self.downscale = downscale
        self.flatten = flatten
        self.render = render

        # if self.grayscale:
        #     # TODO: it is only applicable if the environment is Atari games.
        #     self.env.env._get_image = self.env.env.ale.getScreenGrayscale

        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.atari = True
        else:
            self.atari = False
            
        if self.atari:
            # Atari games
            MDP.__init__(self, range(self.env.action_space.n), self._transition_func, self._reward_func, init_state=GymState(self.process_state(self.env.reset())))
        else:
            # MuJoCo experiments
            MDP.__init__(self, self.env.action_space, self._transition_func, self._reward_func, init_state=GymState(self.env.reset()))

    def process_state(self, state):
        # print('state.shape=', state.shape)
        if self.atari:
            if self.downscale:
                # print('state.shape=', state.shape)
                if self.grayscale:
                    # TODO: For whatever reason imresize require to be 2d-image in this case.
                    state = np.reshape(state, (210, 160))
                    obs = imresize(state, (105, 80), interp='nearest')
                    # obs = cv.resize(state, (105, 80), interpolation=cv.INTER_NEAREST)
                else:
                    # We use nearest to avoid having new color in the screen.
                    s = state.astype(np.uint8)
                    # s = np.reshape((210, 160, 3))
                    # print('s.shape=', s.shape)
                    # print('state.shape=', state.shape)
                    obs = imresize(s, (105, 80, 3), interp='nearest')
                    # obs = imresize(state, (105, 80, 3))
                
                    # obs = np.zeros((105, 80, 3))
                    # for i in range(3):
                    #     rgb = state[:, :, i]
                    #     rgb_s = cv.resize(state, (80, 105), interpolation=cv.INTER_NEAREST)
                    #     obs[:, :, i] = rgb_s
            else:
                obs = state
        else:
            obs = state
        if self.flatten:
            obs = obs.flatten()
        return obs
            
    def get_parameters(self):
        '''
        Returns:
            (dict) key=param_name (str) --> val=param_val (object).
        '''
        param_dict = defaultdict(int)
        param_dict["env_name"] = self.env_name
   
        return param_dict

    def _reward_func(self, state, action):
        '''
x        Args:
            state (AtariState)
            action (str)

        Returns
            (float)
        '''
        obs, reward, is_terminal, self.next_info = self.env.step(action)

        obs_f = self.process_state(obs)
        
        # for i, v in enumerate(obs_f):
        #     if v > 0:
        #         print('obs[', i, '] = ', v)
        # print('obs=', obs)

        # print('next_info=', self.next_info)

        # TODO: Hack to make MontezumaRevenge terminates with 1 life.
        # if 'ale.lives' in self.next_info and self.next_info['ale.lives'] == 5:
        #     is_terminal = True
        
        if self.render:
            self.env.render()

        self.next_state = GymState(obs_f, is_terminal=is_terminal)

        # print('next_state.data=', self.next_state.data)

        if type(reward) is np.ndarray:
            return reward[0]
        return reward

    def _transition_func(self, state, action):
        '''
        Args:
            state (AtariState)
            action (str)

        Returns
            (State)
        '''
        return self.next_state

    def get_image(self):
        # MuJoCo only
        view = self.env.render(mode="rgb_array")
        return view

    def bounds(self):
        # Low -> High
        low = self.env.observation_space.low.flatten()
        high = self.env.observation_space.high.flatten()
        return low, high

    def action_dim(self):
        return self.env.action_space.shape[0]

    def action_bounds(self):
        low = self.env.action_space.low.flatten()
        high = self.env.action_space.high.flatten()
        return low, high

    def reset(self):        
        self.init_state = copy.deepcopy(GymState(self.process_state(self.env.reset()), False))
        self.next_state = None
        self.next_info = None
        self.cur_state = copy.deepcopy(self.init_state)

    def __str__(self):
        return "gym-" + str(self.env_name)
