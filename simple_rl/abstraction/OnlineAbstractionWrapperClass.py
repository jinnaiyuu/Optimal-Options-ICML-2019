# Python imports.
import os
import numpy as np

# Other imports.
from simple_rl.agents import Agent, RMaxAgent, FixedPolicyAgent
from simple_rl.abstraction.state_abs.StateAbstractionClass import StateAbstraction
from simple_rl.abstraction.action_abs.ActionAbstractionClass import ActionAbstraction
from simple_rl.abstraction import aa_helpers

# Options
from options.option_generation.fiedler_options import FiedlerOptions
from options.option_generation.eigenoptions import Eigenoptions
from options.option_generation.betweenness_options import BetweennessOptions

class OnlineAbstractionWrapper(Agent):

    def __init__(self,
                    SubAgentClass,
                    agent_params={},
                    state_abstr=None,
                    action_abstr=None,
                    name_ext="-abstr",
                    n_ops=4,
                    freqs=1000,
                    op_n_episodes=10,
                    op_n_steps=100,
                    max_ops=32,
                    method='eigen',
                    mdp=None):
        '''
        Args:
            SubAgentClass (simple_rl.AgentClass)
            agent_params (dict): A dictionary with key=param_name, val=param_value,
                to be given to the constructor for the instance of @SubAgentClass.
            state_abstr (StateAbstraction)
            state_abstr (ActionAbstraction)
            name_ext (str)
        '''

        # Setup the abstracted agent.
        self.agent = SubAgentClass(**agent_params)
        self.action_abstr = action_abstr
        self.state_abstr = state_abstr
        all_actions = self.action_abstr.get_actions() if self.action_abstr is not None else self.agent.actions

        self.n_ops = n_ops
        self.freqs = freqs
        self.op_n_episodes = op_n_episodes
        self.op_n_steps = op_n_steps
        self.max_ops = max_ops
        self.method = method
        self.mdp = mdp
        
        self.transitions = []
        self.n_steps = 0
        self.prev_state = None
        self.cur_n_ops = 0
        
        Agent.__init__(self, name=self.agent.name + name_ext, actions=all_actions)

    def act(self, ground_state, reward):
        '''
        Args:
            ground_state (State)
            reward (float)

        Return:
            (str)
        '''

        if self.state_abstr is not None:
            abstr_state = self.state_abstr.phi(ground_state)
        else:
            abstr_state = ground_state


        if self.action_abstr is not None:
            # TODO: the code doesnt work when the ground state is represented as an array.
            #       I think states should be wrapped as a class object with a standard interface.
            # print('ground_state=', ground_state)
            ground_action = self.action_abstr.act(self.agent, abstr_state, ground_state, reward)
        else:
            ground_action = self.agent.act(abstr_state, reward)


        self.n_steps += 1
        if self.n_steps % self.freqs == 0 and len(self.transitions) > 10 and self.cur_n_ops < self.max_ops:
            options = self.generate_options()
            if len(options) > 0:
                for o in options:
                    self.action_abstr.add_option(o)
                    self.cur_n_ops += 1

        if self.prev_state is not None:
            pair = (self.prev_state, ground_state)
            self.transitions.append(pair)
        self.prev_state = ground_state
        
        return ground_action

    def generate_options(self):
        # TODO: Train the policy using the experience replay buffer instead of samling new trajectories.
        
        A, intToS = self.generate_matrix()
        known_region = list(intToS.values())
        if self.method == 'eigen':
            # TODO: how is A represented?
            # print('matrix= ', A)
            _, options, vectors = Eigenoptions(A, self.n_ops)
        elif self.method == 'fiedler':
            _, options, _, vectors = FiedlerOptions(A, self.n_ops)
        elif self.method == 'bet':
            _, options, vectors = BetweennessOptions(A, self.n_ops)
        else:
            assert(False)

        print('generated options: ')
        for i, o in enumerate(options):
            if type(o[0]) is list:
                print('inits:')
                for ss in o[0]:
                    print(intToS[ss])
                print('goals:')
                for ss in o[1]:
                    print(intToS[ss])
            else:
                print('init:', intToS[o[0]])
                print('goal:', intToS[o[1]])
                    
        egoal_list = [[]] * (len(options) * 2) 
        for i, o in enumerate(options):
            if type(o[0]) is list:
                for ss in o[0]:
                    egoal_list[i * 2].append(intToS[ss])
                for ss in o[1]:
                    egoal_list[i * 2 + 1].append(intToS[ss])
            else:
                egoal_list[i * 2] = [intToS[o[0]]]
                egoal_list[i * 2 + 1] = [intToS[o[1]]]

        evector_list = [dict()] * (len(options) * 2)
        for i, o in enumerate(options):
            for j in intToS.keys():
                # print('hash(', j, ')=', hash(intToS[j]))
                # print('s[j]=', intToS[j])
                # for i in intToS[j].data.flatten():
                #     if i > 0:
                #         print(i)
                evector_list[i * 2][hash(intToS[j])] = -vectors[i][j]
                evector_list[i * 2 + 1][hash(intToS[j])] = vectors[i][j]

        # TODO: policy is computed using vi right now.
        goal_options = aa_helpers.make_subgoal_options(self.mdp, egoal_list, known_region, vectors=evector_list, n_trajs=self.op_n_episodes, n_steps=self.op_n_steps, classifier='list', policy='vi')

        return goal_options

    def generate_matrix(self):
        # TODO: Add edges corresponding to the options already generated.

        # Similar to util.getIncidenceMatrix
        hash_to_ind = dict() # Dictionary of state -> index
        ind_to_s = dict()

        n_states = 0
        
        for pair in self.transitions:
            for p in pair:
                next_h = hash(p)
                if next_h in hash_to_ind.keys():
                    next_i = hash_to_ind[next_h]
                else:
                    next_i = n_states
                    hash_to_ind[next_h] = next_i
                    ind_to_s[next_i] = p
                    n_states += 1
            
        # n_states = len(self.transitions)

        matrix = np.zeros((n_states, n_states), dtype=int)

        for pair in self.transitions:
            if not hash(pair[0]) == hash(pair[1]):
                matrix[hash_to_ind[hash(pair[0])], hash_to_ind[hash(pair[1])]] += 1 # TODO: plus 1?
         
        # print('n_states=', n_states)
        # print('matrix.shape=', matrix.shape)
        # print('matrix=', matrix)

        return matrix, ind_to_s


    def reset(self):
        # Write data.

        n_ops = len(self.action_abstr.options)
        n_prims = n_ops - self.cur_n_ops
        self.action_abstr.options = self.action_abstr.options[0:n_prims]
        
        self.transitions = []
        self.n_steps = 0
        self.prev_state = None
        self.cur_n_ops = 0

        

        self.agent.reset()

        if self.action_abstr is not None:

            self.action_abstr.reset()

    def end_of_episode(self):
        self.agent.end_of_episode()
        if self.action_abstr is not None:
            self.action_abstr.end_of_episode()
