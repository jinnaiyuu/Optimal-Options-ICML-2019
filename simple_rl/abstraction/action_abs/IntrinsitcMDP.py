import gym
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.gym.GymStateClass import GymState
from collections import defaultdict
import copy

class IntrinsicMDP(MDP):
    '''
    GymMDP with intrinsitc reward added
    '''
    
    def __init__(self, intrinsic_reward, env_name=None, mdp=None):
        '''
        Intrinsic reward should be a function (GymState -> float).
        '''
        self.intrinsic_reward = intrinsic_reward

        if mdp is not None:
            self.env = copy.deepcopy(mdp)
            MDP.__init__(self, mdp.actions, self._transition_func, self._reward_func, init_state=mdp.get_init_state())
        else:
            self.env_name = env_name
            self.env = gym.make(env_name)
            MDP.__init__(self, range(self.env.action_space.n), self._transition_func, self._reward_func, init_state=MDPState(self.env.reset()))

        

        self.next_state = self.get_init_state()

        
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
        Args:
            state (AtariState)
            action (str)

        Returns
            (float)
        '''
        reward = self.env._reward_func(state, action)

        cur_state = self.env.get_curr_state()

        if hash(self.next_state) in self.intrinsic_reward.keys():
            next_in_r = self.intrinsic_reward[hash(self.next_state)]
        else:
            next_in_r = 0
        
        if hash(cur_state) in self.intrinsic_reward.keys():
            cur_in_r = self.intrinsic_reward[hash(cur_state)]
        else:
            cur_in_r = 0
        
        return reward + next_in_r - cur_in_r

    def _transition_func(self, state, action):
        '''
        Args:
            state (AtariState)
            action (str)

        Returns
            (State)
        '''
        next_state = self.env._transition_func(state, action)
        return next_state

    def reset(self):
        self.env.reset()

    def __str__(self):
        return "gym-" + str(self.env_name)
