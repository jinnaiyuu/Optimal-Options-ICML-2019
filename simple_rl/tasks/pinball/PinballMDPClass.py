# Python Imports.
from __future__ import print_function
import copy
import numpy as np
import os

# Other imports.
from rlpy.Domains.Pinball import Pinball
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.pinball.PinballStateClass import PinballState
import pdb
import numpy as np

# import pygame

from rlpy.Tools import __rlpy_location__

class PinballMDP(MDP):
    """ Class for pinball domain. """
    # TODO: Can we implement a pygame-based visualizer?
    #       rlpy is based on Tkinter.

    def __init__(self, noise=0., episode_length=1000, reward_scale=1000., cfg="pinball_empty.cfg", render=False):
        # default_config_dir = os.path.join(__rlpy_location__, "Domains", "PinballConfigs")
        default_config_dir = os.path.dirname(__file__)
        self.cfg = cfg
        self.domain = Pinball(noise=noise, episodeCap=episode_length, configuration=os.path.join(default_config_dir, "PinballConfigs", self.cfg))
        self.render = render
        self.reward_scale = reward_scale

        # Each observation from domain.step(action) is a tuple of the form reward, next_state, is_term, possible_actions
        # s0 returns initial state, is_terminal, possible_actions
        init_observation = self.domain.s0()
        init_state = tuple(init_observation[0])

        actions = self.domain.actions

        MDP.__init__(self, actions, self._transition_func, self._reward_func, init_state=PinballState(*init_state))

    def get_parameters(self):
        '''
        Returns:
            (dict) key=param_name (str) --> val=param_val (object).
        '''
        param_dict = {}
        param_dict["gamma"] = self.gamma
        param_dict["step_cost"] = self.step_cost
        param_dict["cfg"] = self.cfg
        return param_dict

        
    def _reward_func(self, state, action, option_idx=None):
        """
        Args:
            state (PinballState)
            action (int): number between 0 and 4 inclusive
            option_idx (int): was the action selected based on an option policy

        Returns:
            next_state (PinballState)
        """
        assert self.is_primitive_action(action), "Can only implement primitive actions to the MDP"
        reward, obs, done, possible_actions = self.domain.step(action)

        if self.render:
            self.domain.showDomain(action)
            # self.domain.showDomain(action, option_idx)

        self.next_state = PinballState(*tuple(obs), is_terminal=done)

        assert done == self.is_goal_state(self.next_state), "done = {}, s' = {} should match".format(done, self.next_state)

        negatively_clamped_reward = -1. if reward < 0 else reward
        return negatively_clamped_reward / self.reward_scale


    def _transition_func(self, state, action):
        return self.next_state

    def execute_agent_action(self, action, option_idx=None):
        """
        Args:
            action (str)
            option_idx (int): given if action came from an option policy

        Returns:
            (tuple: <float,State>): reward, State

        Summary:
            Core method of all of simple_rl. Facilitates interaction
            between the MDP and an agent.
        """
        reward = self.reward_func(self.cur_state, action, option_idx)
        next_state = self.transition_func(self.cur_state, action)
        self.cur_state = next_state

        return reward, next_state

    def reset(self):
        init_observation = self.domain.s0()
        init_state = tuple(init_observation[0])
        self.init_state = PinballState(*init_state)
        cur_state = copy.deepcopy(self.init_state)
        self.set_current_state(cur_state)

    def set_current_state(self, new_state):
        self.cur_state = new_state
        self.domain.state = new_state.features()

    def is_goal_state(self, state):
        """
        We will pass a reference to the PinballModel function that indicates
        when the ball hits its target.
        Returns:
            is_goal (bool)
        """
        target_pos = np.array(self.domain.environment.target_pos)
        target_rad = self.domain.environment.target_rad
        return np.linalg.norm(state.get_position() - target_pos) < target_rad

    def bounds(self):
        # Low and then high
        low_bound = np.asarray([0.0, 0.0, -2.0, -2.0])
        up_bound = np.asarray([1.0, 1.0, 2.0, 2.0])
        return low_bound, up_bound

    # def draw_state(self, s, filename=None):
    #     draw_trajectory([s], filename=filename)
    # 
    # def draw_trajectory(self, traj, filename=None):
    #     assert(isinstance(traj, list))
    #     # 1. set up canvas of size 500x500.
    #     width = 1000
    #     height = 1000
    #     screen = pygame.Surface((width, height))
    #     screen.fill((255, 255, 255))
    # 
    #     # 2. Render the obstacle positions.
    #     for obs in self.domain.environment.obstacles:
    #         point_list = obs.points
    #         plist = []
    #         for p in point_list:
    #             pair = (int(p[0] * width), int(p[1] * height))
    #             plist.append(pair)
    #         # print('point_list=', point_list)
    #         # TODO: Normalize to the width/height
    #         pygame.draw.polygon(screen, (0, 0, 0), plist, 0)    
    #     
    #     # 3. Render the trajectories
    #     # print('traj=', traj)
    #     lines = []
    #     for i in range(len(traj)):
    #         lines.append((int(width * traj[i].x), int(height * traj[i].y)))
    #     pygame.draw.lines(screen, (46, 224, 49), False, lines, 15)
    #     # print('lines=', lines)
    #     # ball_pos = (int(s[0] * width), int(s[1] * height))
    #     pygame.draw.circle(screen, (46, 224, 224), lines[0], 15)
    #     pygame.draw.circle(screen, (224, 145, 157), lines[-1], 15)
    # 
    #     # 4. Render the goal position
    #     # target_pos = self.domain.environment.target_pos
    #     # tpos = (int(target_pos[0] * width), int(target_pos[1] * height))
    #     # TODO: Normalize to the width/height
    #     # pygame.draw.circle(screen, (224, 0, 0), tpos, 15)
    # 
    #     flipped_scr = pygame.transform.flip(screen, False, True)
    #     # 5. Print to file.
    #     if filename is None:
    #         pygame.image.save(flipped_scr, './pinball_state.png')
    #     else:
    #         pygame.image.save(flipped_scr, filename)


    @staticmethod
    def is_primitive_action(action):
        assert action >= 0, "Action cannot be negative {}".format(action)
        return action < 5

    def __str__(self):
        return "RlPy_Pinball_Domain"

    def __repr__(self):
        return str(self)

