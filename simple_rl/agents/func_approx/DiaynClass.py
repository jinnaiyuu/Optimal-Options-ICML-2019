# Q learning with NN
import numpy as np
import tensorflow as tf
from copy import copy
from random import choice, random
import math
import datetime
import re
import tflearn

from .util import leakyReLU, logClp, logClpNp, one_hot

from simple_rl.agents.AgentClass import Agent
from simple_rl.agents.func_approx.ExperienceBuffer import ExperienceBuffer
from simple_rl.agents.func_approx.ActorNetwork import Actor
from simple_rl.agents.func_approx.CriticNetwork import Critic
from simple_rl.mdp.StateClass import State

class DiaynAgent(Agent):
    """
    Diversity is all you need
    """

    # Parameters
    hidden_size = 128

    NAME = "Diayn"

    def __init__(self, sess=None, obs_dim=None, num_actions=None, action_dim=None, action_bound=None, num_options=None, batch_size=64, buffer_size=512, update_freq=32, alpha=0.1, lowmethod='nn', gamma=0.99, name=NAME):
        Agent.__init__(self, name=name, actions=[])
        
        self.obs_dim = obs_dim
        self.num_actions = num_actions # TODO: we want to take continuous action-space too.
        self.action_dim = action_dim
        self.action_bound = action_bound
        
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.update_freq = update_freq

        self.lowmethod = lowmethod
        self.gamma = gamma
        
        self.alpha = alpha
        self.num_options = num_options

        self.pz = np.full(self.num_options, 1.0/self.num_options) # Skill distribution
        self.current_skill = None

        if sess is None:
            self.sess = tf.Session()
        else:
            self.sess = sess

        self.train_diversity = True

        if self.num_actions is not None:
            # Discrete action space
            self.action_dim = self.num_actions
            bound = (-np.ones(self.num_actions), np.ones(self.num_actions))
            self.actor = Actor(self.sess, obs_dim=self.obs_dim + self.num_options, action_dim=self.action_dim, action_bound=bound, learning_rate=0.005, batch_size=self.batch_size, tau=0.1, lowmethod=self.lowmethod, name='diayn_actor')
        else:
            self.actor = Actor(self.sess, obs_dim=self.obs_dim + self.num_options, action_dim=action_dim, action_bound=action_bound, learning_rate=0.005, batch_size=self.batch_size, tau=0.1, lowmethod=self.lowmethod, name='diayn_actor')
            
        if self.num_actions is not None:
            self.critic = Critic(sess=self.sess, obs_dim=self.obs_dim + self.num_options, action_dim=self.num_actions, learning_rate=0.005, tau=0.1, lowmethod=self.lowmethod, name=self.name + '_critic')
        else:
            self.critic = Critic(sess=self.sess, obs_dim=self.obs_dim + self.num_options, action_dim=self.action_dim, learning_rate=0.005, tau=0.1, lowmethod=self.lowmethod, name=self.name + '_critic')

        self.discriminator = Discriminator(sess=self.sess, obs_dim=self.obs_dim, num_options=self.num_options, hidden_size=200, learning_rate=0.00005, name=self.name + '_discrim')

        self.reset()
        
    def sample_skill(self):
        return np.random.choice(self.num_options, p=self.pz)
        
    def act(self, state, reward, learning=True):
        if self.experience_buffer.size() > self.batch_size and self.total_steps % self.update_freq == 0 and learning:
            s, a, r, s2, t, o = self.experience_buffer.sample_op(self.batch_size)
            self.train_batch(s, a, r, s2, t, o, batch_size=self.batch_size)
        
        if self.current_skill is None:
            self.current_skill = self.sample_skill()
        # if random() < self.epsilon:
        #     # print("random")
        #     return choice(range(self.num_actions))
        # else:

        skill_onehot = one_hot(self.current_skill, self.num_options, 1.0, 0.0)   
        state_skl = np.concatenate([state.data, skill_onehot]) 
        

        if self.num_actions is not None:
            # assert(False)
            prob_action = self.actor.predict_discrete(np.reshape(state_skl, (1, self.obs_dim + self.num_options)))[0]
            # print('prob_action=', prob_action)
            # print('sum=', sum(prob_action))
            action = np.random.choice(range(self.num_actions), p=prob_action)
        else:
            # High-dimensional action space
            action = self.actor.predict(np.reshape(state_skl, (1, self.obs_dim + self.num_options)))[0]

        if not (self.prev_state is None) and not (self.prev_action is None) and learning:
            self.experience_buffer.add([self.prev_state, self.prev_action, reward, state, state.is_terminal(), self.current_skill])


        self.prev_state, self.prev_action = state, action

        self.total_steps += 1

        return action

        
    def train_batch(self, s, a, r, s2, t, o, batch_size):
        
        s_tnsr = np.asarray(self.convert_states_to_arrays(s))
        s2_tnsr = np.asarray(self.convert_states_to_arrays(s2))

        o_oh_ = []
        for i in range(len(o)):
            skill_onehot = one_hot(o[i], self.num_options)
            o_oh_.append(skill_onehot)
        o_oh = np.asarray(o_oh_)
        # print('o_ohshaep=', o_oh.shape)
        # skill_onehot = np.reshape(skill_onehot_, (batch_size, self.num_options))

        # TODO: Because the option never terminates, it should be fine to assume that the option applied to s2 is the same as the option on s.
        # print('s_tnsr=', s_tnsr.shape)
        # print('skill_onehot=', o_oh.shape)

        so_tnsr = np.concatenate((s_tnsr, o_oh), 1)
        so2_tnsr = np.concatenate((s2_tnsr, o_oh), 1)
        
        so_q = self.critic.predict_target(so_tnsr, np.reshape(self.actor.predict_target(so_tnsr), (batch_size, self.action_dim)))
        so2_q = self.critic.predict_target(so2_tnsr, np.reshape(self.actor.predict_target(so2_tnsr), (batch_size, self.action_dim)))


        if self.train_diversity:
            self.discriminator.train(s2_tnsr, o_oh)
            pz_ = []
            for i in range(batch_size):
                pz_.append(self.pz[o[i]])
            pz = np.asarray(pz_)
                
            # pz = self.pz[o]
            # print('pz=', pz)
            pz_s = self.discriminator.predict(s2_tnsr)
            diversity_reward = logClpNp(np.sum(pz_s * o_oh, axis=1)) - logClpNp(pz)

            advantages = np.zeros(batch_size)
            for i in range(len(advantages)):
                advantages[i] = diversity_reward[i] + self.gamma * so2_q[i] - so_q[i]
        else:
            advantages = np.zeros(batch_size)
            for i in range(len(advantages)):
                advantages[i] = r[i] + self.gamma * so2_q[i] - so_q[i]


        if self.num_actions is not None:
            a_oh_ = []
            for i in range(len(a)):
                skill_onehot = one_hot(a[i], self.num_actions)
                a_oh_.append(skill_onehot)
            a_oh = np.asarray(a_oh_)
            _, _ = self.critic.train(so_tnsr, a_oh, np.reshape(advantages, (batch_size, 1)))
        else:
            _, _ = self.critic.train(so_tnsr, a, np.reshape(advantages, (batch_size, 1)))
            

        a_outs = self.actor.predict(so_tnsr)
        grads = self.critic.action_gradients(so_tnsr, a_outs)
        self.actor.train(so_tnsr, grads[0])
        
        self.actor.update_target_network()
        self.critic.update_target_network()        

    def end_of_episode(self):
        self.current_skill = None

        self.prev_state, self.prev_action = None, None
        

    def reset(self):
        self.experience_buffer = ExperienceBuffer(buffer_size=self.buffer_size)
        self.prev_state, self.prev_action = None, None
        self.curr_step, self.total_steps = 0, 0
        self.curr_episode = 0

        # Initialize the network

        self.actor.initialize()
        self.critic.initialize()
        self.discriminator.initialize()

    def convert_states_to_arrays(self, obs):
        assert(isinstance(obs, list))

        arrays = []
        for s in obs:
            assert(isinstance(s, State))
            arrays.append(s.data)

        # tnsr = np.stack(arrays, axis=0)
        return arrays

    def set_diversity(self, val):
        assert(isinstance(val, bool))
        self.train_diversity = val

    def restore(self, directory, name=None):
        if name is None:
            name = self.name
            
        self.critic.restore(directory, name + '_critic')
        self.actor.restore(directory, name + '_actor')
                        
    def save(self, directory, name=None):
        if name is None:
            name = self.name
        self.critic.save(directory, name + '_critic')
        self.actor.save(directory, name + '_actor')



class Discriminator(object):
    def __init__(self, sess, obs_dim, num_options, hidden_size, learning_rate, name='discrim'):
        self.sess = sess
        self.obs_dim = obs_dim
        self.num_options = num_options
        self.learning_rate = learning_rate

        self.hidden_size = hidden_size

        self.name = name

        self.obs, self.pz_s = self.discrim_network(scope=name)
        self.skill = tf.placeholder(tf.float32, [None, num_options], name='skill') # one-hot representation
        self.loss = tf.reduce_sum(logClp(self.pz_s) * self.skill)

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.optimize = self.optimizer.minimize(self.loss)

        self.network_params = tf.trainable_variables(scope=name)
        self.initializer = tf.initializers.variables(self.network_params + self.optimizer.variables())

        self.saver = tf.train.Saver(self.network_params + self.optimizer.variables())
        
    def discrim_network(self, scope):
        """
        Given a state, infer a skill which led the agent to the state
        """
        obs = tflearn.input_data(shape=[None, self.obs_dim], name='obs')
        
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            Ws = tf.Variable(tf.truncated_normal([self.obs_dim, self.hidden_size], dtype=tf.float32, stddev=0.1))
            b1 = tf.Variable(tf.truncated_normal([self.hidden_size], dtype=tf.float32, stddev=0.0001))
            W2 = tf.Variable(tf.truncated_normal([self.hidden_size, self.hidden_size], dtype=tf.float32, stddev=0.1))
            b2 = tf.Variable(tf.truncated_normal([self.hidden_size], dtype=tf.float32, stddev=0.1))
            W3 = tf.Variable(tf.truncated_normal([self.hidden_size, self.num_options], dtype=tf.float32, stddev=0.1))
            b3 = tf.Variable(tf.truncated_normal([self.num_options], dtype=tf.float32, stddev=0.1))

            layer_1 = leakyReLU(tf.matmul(obs, Ws) + b1)
            layer_2 = leakyReLU(tf.add(tf.matmul(layer_1, W2), b2))
            layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, W3), b3))
            pz_s = tf.nn.softmax(layer_3)

        return obs, pz_s

    def train(self, obs, skill):
        # print('skill.shape=', skill.shape)
        return self.sess.run([self.pz_s, self.optimize], feed_dict={
            self.obs: obs,
            self.skill: skill
        })

    def predict(self, obs):
        return self.sess.run(self.pz_s, feed_dict={
            self.obs: obs
        })

    def initialize(self):
        return self.sess.run(self.initializer)

    def restore(self, directory, name=None):
        if name is None:
            self.saver.restore(self.sess, directory + '/' + self.name)
        else:
            self.saver.restore(self.sess, directory + '/' + name)
    
    def save(self, directory, name=None):
        if name is None:
            self.saver.save(self.sess, directory + '/' + self.name)
        else:
            self.saver.save(self.sess, directory + '/' + name)
