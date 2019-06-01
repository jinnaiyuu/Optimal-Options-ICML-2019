'''
LinearQLearningAgentClass.py

Contains implementation for a Q Learner with a Linear Function Approximator.
'''

# Python imports.
import numpy as np
import math
from collections import defaultdict

# Other imports.
from simple_rl.agents import Agent, QLearningAgent
# from simple_rl.agents.agents.Features import RBF, Fourier

class LinearQAgent(QLearningAgent):
    '''
    QLearningAgent with a linear function approximator for the Q Function.
    '''

    def __init__(self, actions, rand_init=True, name="Linear-Q", alpha=0.001, gamma=0.99, epsilon=0.2, explore="uniform", feature=None, anneal=True, sarsa=False):
        # name = name + "-rbf" if rbf else name
        QLearningAgent.__init__(self, actions=list(actions), name=name, alpha=alpha, gamma=gamma, epsilon=epsilon, explore=explore, anneal=anneal)

        self.sarsa = sarsa

        self.feature = feature # function (state, action) -> features (numpy vector)
        
        self.num_features = self.feature.num_features()
        # Add a basis feature.
        self.rand_init = rand_init
        if rand_init:
            self.weights = np.random.random(self.num_features*len(self.actions))
        else:
            self.weights = np.zeros(self.num_features*len(self.actions))


        self.max_weight = 0.0
        # self.rbf = rbf
        # self.fourier = fourier # fourier is the order of the Fourier basis.
        # self.coeff = None

        

    def get_parameters(self):
        '''
        Returns:
            (dict) key=param_name (str) --> val=param_val (object).
        '''
        param_dict = defaultdict(int)
        
        param_dict["num_features"] = self.num_features
        param_dict["rand_init"] = self.rand_init
        param_dict["alpha"] = self.alpha
        param_dict["gamma"] = self.gamma
        param_dict["epsilon"] = self.epsilon
        # param_dict["rbf"] = self.rbf
        param_dict["anneal"] = self.anneal
        param_dict["explore"] = self.explore

        return param_dict

    def update(self, state, action, reward, next_state):
        '''
        Args:
            state (State)
            action (str)
            reward (float)
            next_state (State)

        Summary:
            Updates the internal Q Function according to the Bellman Equation. (Classic Q Learning update)
        '''
        if state is None:
            # If this is the first state, initialize state-relevant data and return.
            self.prev_state = state
            return
        self._update_weights(reward, next_state)

    def _phi(self, state, action):
        '''
        Args:
            state (State): The abstract state object.
            action (str): A string representing an action.

        Returns:
            (numpy array): A state-action feature vector representing the current State and action.

        Notes:
            The resulting feature vector multiplies the state vector by |A| (size of action space), and only the action passed in retains
            the original vector, all other values are set to 0.
        '''
        result = np.zeros(self.num_features * len(self.actions))
        act_index = self.actions.index(action)

        basis_feats = state.features()

        if self.feature is not None:
            basis_feats = self.feature.feature(state, action)

        result[act_index*self.num_features:(act_index + 1)*self.num_features] = basis_feats

        # Check if the features are set to numbers
        assert(not np.isnan(np.sum(result)))

        return result

    def _update_weights(self, reward, cur_state):
        '''
        Args:
            reward (float)
            cur_state (State)

        Summary:
            Updates according to:

            [Eq. 1] delta = r + gamma * max_b(Q(s_curr,b)) - Q(s_prev, a_prev)

            For each weight:
                w_i = w_i + alpha * phi(s,a)[i] * delta

            Where phi(s,a) maps the state action pair to a feature vector (see QLearningAgent._phi(s,a))
        '''

        # Compute temporal difference [Eq. 1]
        if self.sarsa:
            next_action = self.epsilon_greedy_q_policy(cur_state)
            max_q_cur_state = self.get_q_value(cur_state, next_action)
        else:
            max_q_cur_state = self.get_max_q_value(cur_state)
        prev_q_val = self.get_q_value(self.prev_state, self.prev_action)

        # If the Q(s, a) is smaller than R(s, a) + y * V(s'), then we increase the value of Q(s, a).
        self.most_recent_loss = reward + self.gamma * max_q_cur_state - prev_q_val # TD error

        # Update each weight
        phi = self._phi(self.prev_state, self.prev_action)
        
        
        active_feats_index = self.actions.index(self.prev_action) * self.num_features

        # Sparsely update the weights (only update weights associated with the action we used).
        max_weight = 0.0
        for i in range(active_feats_index, active_feats_index + self.num_features):
            # Multiply by the norm?
            # self.weights[i] = self.weights[i] + self.alpha * phi[i] * self.most_recent_loss
            self.weights[i] = self.weights[i] + self.alpha * phi[i] * self.most_recent_loss
            
            # TODO: This hack is not appropriate. Should be removed later on once I found the problem.
            if abs(self.weights[i]) > max_weight:
                max_weight = abs(self.weights[i])

        if max_weight > self.max_weight:
            # print('max weight=', max_weight)
            self.max_weight = max_weight

        # Check if the weights are set to numbers
        assert(not np.isnan(np.sum(self.weights)))

    def get_q_value(self, state, action):
        '''
        Args:
            state (State): A State object containing the abstract state representation
            action (str): A string representing an action. See namespaceAIX.

        Returns:
            (float): denoting the q value of the (@state,@action) pair.
        '''

        # Return linear approximation of Q value
        sa_feats = self._phi(state, action)

        ret =  np.dot(self.weights, sa_feats)
        assert(isinstance(ret, float))
        assert(not np.isnan(ret))
        return ret


    def train_batch(self, s, a, r, s2, t, duration=None, batch_size=1):
        # print('training agent: ', self.name)
        if duration is None:
            duration = [1] * len(s)
        # For primitive actions, duration = 1

        # TODO: Do we update the weight for each sample? Or do we update once for a batch?

        total_loss = 0.0
        for i in range(batch_size):
            # print('s=', s[i])
            # print('a=', a[i])
            # print('r=', r[i])
            # print('s2=', s2[i])
            # print('dur=', duration[i])
            # 
            # print('F(s) =', self.feature.feature(s[i], 0))
            # print('F(s2)=', self.feature.feature(s2[i], 0))

            if self.sarsa:
                next_action = self.epsilon_greedy_q_policy(s[i])
                max_q_cur_state = self.get_q_value(s2[i], next_action)
            else:
                max_q_cur_state = self.get_max_q_value(s2[i])
                
            prev_q_val = self.get_q_value(s[i], a[i])
            most_recent_loss = r[i] + pow(self.gamma, duration[i]) * max_q_cur_state - prev_q_val
            assert(not np.isnan(most_recent_loss))
            total_loss += most_recent_loss
            # print('Q(s, a)   estm=', prev_q_val)
            # print('Backup value=', r[i] + pow(self.gamma, duration[i]) * max_q_cur_state)
            # print('Q(s\', a\') estm=', max_q_cur_state)
            # print('r=', r[i])
            
            
            # Update each weight
            phi = self._phi(s[i], a[i])
            active_feats_index = self.actions.index(a[i]) * self.num_features


            max_weight = 0.0
            # Sparsely update the weights (only update weights associated with the action we used).
            for j in range(active_feats_index, active_feats_index + self.num_features):
                self.weights[j] = self.weights[j] + self.alpha * phi[j] * most_recent_loss / float(batch_size)

                # TODO: This hack is not appropriate. Should be removed later on once I found the problem.
                if self.weights[j] > 1.0:
                    if abs(self.weights[j]) > max_weight:
                        max_weight = abs(self.weights[j])
                    # print('weight=', self.weights[j])
                    # self.weights[j] = 1.0
                elif self.weights[j] < -1.0:
                    if abs(self.weights[j]) > max_weight:
                        max_weight = abs(self.weights[j])
                    
                    # print('weight=', self.weights[j])
                    # self.weights[j] = -1.0

            if max_weight > self.max_weight:
                # print('max weight=', max_weight)
                self.max_weight = max_weight
                
            # Check if the weights are set to numbers
            if np.isnan(np.sum(self.weights)):
                print('')
                print('weights get to NaN!!!!!!!!')
                print('max_q_cur_state=', max_q_cur_state)
                print('prev_q_val=', prev_q_val)
                print('Backup value: r + y V(s\') =', r[i] + pow(self.gamma, duration[i]) * max_q_cur_state)
                print('r=', r[i])
                print('sum(phi)=', np.sum(phi))
                print('loss=', most_recent_loss)
                print('weights are NaN!')
                print('weights=', self.weights)
                print('phi=', phi)
                print('alpha=', self.alpha)
            assert(not np.isnan(np.sum(self.weights)))
        # print('total loss=', total_loss)

    def reset(self):
        # TODO: Optimistic Initialization? 
        self.weights = np.ones(self.num_features*len(self.actions))
        QLearningAgent.reset(self)


    def restore(self, directory, rev=False, name='linearq'):
        if rev:
            self.weights = np.load(directory + '/' + name + 'rev.npy')
        else:
            self.weights = np.load(directory + '/' + name + '.npy')            
            
    def save(self, directory, name='linearq'):
        if rev:
            np.save(directory + '/' + name + 'rev.npy', self.weights)
        else:
            np.save(directory + '/' + name + '.npy', self.weights)
