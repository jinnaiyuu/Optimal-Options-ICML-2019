# Q learning with NN
import numpy as np
import tensorflow as tf
from copy import copy
from random import choice, random
import math
import datetime
import re

from simple_rl.agents.AgentClass import Agent
from simple_rl.agents.func_approx.ExperienceBuffer import ExperienceBuffer
from .util import leakyReLU, logClp

class DiaynAgent(Agent):
    """
    Diversity is all you need
    """

    # Parameters
    hidden_size = 128

    NAME = "Diayn"

    def __init__(self, sess=None, obs_dim=None, num_actions=None, action_dim=None, action_bound=None, num_options=None, batch_size=64, buffer_size=512, update_freq=32, alpha=0.1, gamma=0.99, name=NAME):
        Agent.__init__(self, name=name, actions=[])
        
        self.obs_dim = obs_dim
        self.num_actions = num_actions # TODO: we want to take continuous action-space too.
        self.action_dim = action_dim
        self.action_bound = action_bound
        
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.update_freq = update_freq

        self.gamma = gamma
        
        self.alpha = alpha
        self.num_options = num_options

        self.pz = np.full(self.num_options, 1.0/self.num_options) # Skill distribution
        self.current_skill = None

        if sess is None:
            self.sess = tf.Session()
        else:
            self.sess = sess

        self.prev_states = tf.placeholder(shape=[None, obs_dim], dtype=tf.float32, name="prev_states")
        self.rewards = tf.placeholder(shape=[None], dtype=tf.float32, name="rewards")
        self.entropies = tf.placeholder(shape=[None], dtype=tf.float32, name="entripies")

        # TODO: States and actions have to be able to take continuous values.
        
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")
        self.states = tf.placeholder(shape=[None, obs_dim], dtype=tf.float32, name="states")
        # self.last_state = tf.placeholder(shape=[1, obs_dim], dtype=tf.float32)
        self.advantages = tf.placeholder(shape=[None], dtype=tf.float32, name="advantages")

        self.skill = tf.placeholder(shape=[None], dtype=tf.int32, name="skill") # We implement a placeholder of a skill as a shape=[None].

        self.pz_tf = tf.placeholder(shape=[None], dtype=tf.float32, name="pz_tf")
        
        self.pa_s, self.Qsa = self.Actor(self.prev_states, self.skill)
        self.bestA = tf.argmax(self.pa_s[0])
        self.actProb = tf.gather(self.pa_s, self.actions, axis=1)

        
        # self.train_act, self.train_crit, self.train_dsc, self.loss, self.actor_loss, self.critic_loss, self.discrimator_loss, self.entropy_out, self.predictedVvalue, self.pz_s, self.diversity_reward =
        self.training(self.prev_states, self.rewards, self.entropies, self.actions, self.skill, self.states, self.advantages, self.pz_tf)

        self.actor_params = tf.get_collection(tf.GraphKeys.VARIABLES, scope=name + "_actor")
        self.critic_params = tf.get_collection(tf.GraphKeys.VARIABLES, scope=name + "_critic")
        self.discrim_params = tf.get_collection(tf.GraphKeys.VARIABLES, scope=name + "_discrim")

        params = self.actor_params + self.critic_params + self.discrim_params
        # params = self.actor_params + self.critic_params + self.discrim_params + self.optimizer_act.variables() + self.optimizer_crit.variables() + self.optimizer_dsc.variables()
        # print('params=', params)
        
        self.initializer = tf.initializers.variables(params + self.optimizer_act.variables() + self.optimizer_crit.variables() + self.optimizer_dsc.variables())

        self.saver = tf.train.Saver(params)

        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("actor loss", self.actor_loss)
        tf.summary.scalar("critic loss", self.critic_loss)
        tf.summary.scalar("discriminator loss", self.discriminator_loss)
        # tf.summary.scalar("entropy loss", self.entropy)
        self.summary = tf.summary.merge_all()

        curTime = re.sub("[^0-9]", "", str(datetime.datetime.now()))
        self.train_writer = tf.summary.FileWriter( './logs/' + curTime, self.sess.graph)
        # self.train_writer = tf.summary.FileWriter( './logs/' + str(curTime), self.sess.graph)

        # TODO: track statistics

        self.reset()
        
    def sample_skill(self):
        return np.random.choice(self.num_options, p=self.pz)

    def Actor(self, prev_states, skill):
        """
        Given a state return a (stochastic) policy: P(a|s).
        """

        # TODO: Implement target function (double networks).
        
        skill_onehot = tf.one_hot(skill, self.num_options, 1.0, 0.0)

        with tf.variable_scope(self.name + "_actor", reuse=tf.AUTO_REUSE):
            W1 = tf.Variable(tf.truncated_normal([self.obs_dim, self.hidden_size], dtype=tf.float32, stddev=0.1))
            Wskl = tf.Variable(tf.truncated_normal([self.num_options, self.hidden_size], dtype=tf.float32, stddev=0.1))
            # b1 = tf.Variable(tf.truncated_normal([self.hidden_size], dtype=tf.float32, stddev=0.1))
            W2 = tf.Variable(tf.truncated_normal([self.hidden_size, self.hidden_size], dtype=tf.float32, stddev=0.1))
            # b2 = tf.Variable(tf.truncated_normal([self.hidden_size], dtype=tf.float32, stddev=0.1))
            W3 = tf.Variable(tf.truncated_normal([self.hidden_size, self.num_actions], dtype=tf.float32, stddev=0.1))
            # b3 = tf.Variable(tf.truncated_normal([self.num_actions], dtype=tf.float32, stddev=0.1))
        
            # layer_1 = leakyReLU(tf.add(tf.matmul(prev_states, W1), b1))
            # layer_2 = leakyReLU(tf.add(tf.matmul(layer_1, W2), b2))
            # pa_s = tf.nn.softmax(tf.add(tf.matmul(layer_2, W3), b3))
            layer_1 = leakyReLU(tf.matmul(prev_states, W1) + tf.matmul(skill_onehot, Wskl))
            layer_2 = leakyReLU(tf.matmul(layer_1, W2))
            Qsa = tf.matmul(layer_2, W3)
            pa_s = tf.nn.softmax(Qsa) # TODO: We cannot take softmax for continuous action space.
        
        return pa_s, Qsa
        
    def Critic(self, states, skill):
        """
        Given a state, return an estimate of Q(s, a) for all a in A.
        """
        skill_onehot = tf.one_hot(skill, self.num_options, 1.0, 0.0)
        # actions_f = tf.one_hot(actions, self.num_actions, 1.0, 0.0)
        with tf.variable_scope(self.name + "_critic", reuse=tf.AUTO_REUSE):
            W1 = tf.Variable(tf.truncated_normal([self.obs_dim, self.hidden_size], dtype=tf.float32, stddev=0.1))
            Wskl = tf.Variable(tf.truncated_normal([self.num_options, self.hidden_size], dtype=tf.float32, stddev=0.1))
            # Wa = tf.Variable(tf.truncated_normal([self.num_actions, self.hidden_size], dtype=tf.float32, stddev=0.1))
            b1 = tf.Variable(tf.truncated_normal([self.hidden_size], dtype=tf.float32, stddev=0.0001))
            W2 = tf.Variable(tf.truncated_normal([self.hidden_size, self.hidden_size], dtype=tf.float32, stddev=0.1))
            b2 = tf.Variable(tf.truncated_normal([self.hidden_size], dtype=tf.float32, stddev=0.1))
            W3 = tf.Variable(tf.truncated_normal([self.hidden_size, 1], dtype=tf.float32, stddev=0.1))
            b3 = tf.Variable(tf.truncated_normal([1], dtype=tf.float32, stddev=0.1))

            layer_1 = leakyReLU(tf.matmul(states, W1) + tf.matmul(skill_onehot, Wskl) + b1)
            layer_2 = leakyReLU(tf.add(tf.matmul(layer_1, W2), b2))
            Vs = tf.add(tf.matmul(layer_2, W3), b3) # We do not use an activation function like ReLU, as Q can be any real-value.
        return Vs[:, 0]

    def Discriminator(self, states):
        """
        Given a state, infer a skill which led the agent to the state
        """
        with tf.variable_scope(self.name + "_discrim", reuse=tf.AUTO_REUSE):
            Ws = tf.Variable(tf.truncated_normal([self.obs_dim, self.hidden_size], dtype=tf.float32, stddev=0.1))
            # Wa = tf.Variable(tf.truncated_normal([self.num_actions, self.hidden_size], dtype=tf.float32, stddev=0.1))
            b1 = tf.Variable(tf.truncated_normal([self.hidden_size], dtype=tf.float32, stddev=0.0001))
            W2 = tf.Variable(tf.truncated_normal([self.hidden_size, self.hidden_size], dtype=tf.float32, stddev=0.1))
            b2 = tf.Variable(tf.truncated_normal([self.hidden_size], dtype=tf.float32, stddev=0.1))
            W3 = tf.Variable(tf.truncated_normal([self.hidden_size, self.num_options], dtype=tf.float32, stddev=0.1))
            b3 = tf.Variable(tf.truncated_normal([self.num_options], dtype=tf.float32, stddev=0.1))

            layer_1 = leakyReLU(tf.matmul(states, Ws) + b1)
            layer_2 = leakyReLU(tf.add(tf.matmul(layer_1, W2), b2))
            layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, W3), b3))
            pz_s = tf.nn.softmax(layer_3) # We do not use an activation function like ReLU, as Q can be any real-value.

        return pz_s
    
    def training(self, prev_states, rewards, entropies, actions, skill, states, advantages, pz_tf):
        # nextQvalue = self.Critic(states)
        # nextVvalue = tf.reduce_max(nextQvalue, axis=1)
        self.pz_s = self.Discriminator(states)
        skill_onehot = tf.one_hot(skill, self.num_options, 1.0, 0.0)
        
        self.diversity_reward = logClp(tf.reduce_sum(self.pz_s * skill_onehot, axis=1)) - logClp(pz_tf) # - log(p_z) gives an intrinsic reward to survive longer.
        
        self.predictedVvalue = self.Critic(prev_states, skill)
        
        loggedActProb = logClp(self.actProb)
        self.entropy_out = -tf.reduce_sum(loggedActProb * self.actProb, 1)
        
        self.actor_loss = -tf.reduce_sum(loggedActProb * advantages) # TODO: for real A2C, this qvalue should be replaced by advantages.

        self.critic_loss = tf.reduce_sum(tf.square(self.diversity_reward + self.alpha * entropies - self.predictedVvalue))

        self.discriminator_loss = tf.reduce_sum(logClp(self.pz_s) * skill_onehot)

        self.loss = self.actor_loss + self.critic_loss + self.discriminator_loss

        # TODO: We need just one optimizer for the sum of the loss, right?

        # self.optimizer = tf.train.AdamOptimizer(learning_rate=0.005)
        # self.optimize = self.optimizer.minimize(self.loss)
        
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        # optimizer_act = tf.train.RMSprop(learning_rate=0.001)
        self.optimizer_act = tf.train.AdamOptimizer(learning_rate=0.005, name="Adam_actor")
        self.optimize_act = self.optimizer_act.minimize(self.actor_loss)

        self.optimizer_crit = tf.train.AdamOptimizer(learning_rate=0.005, name="Adam_critic")
        self.optimize_crit = self.optimizer_crit.minimize(self.critic_loss)

        self.optimizer_dsc = tf.train.AdamOptimizer(learning_rate=0.00005, name="Adam_dsc")
        self.optimize_dsc = self.optimizer_dsc.minimize(self.discriminator_loss)

        # loss = actor_loss + critic_loss + discrimator_loss
        
        # return train_act, train_crit, train_dsc, loss, actor_loss, critic_loss, discrimator_loss, entropy_out, predictedVvalue, pz_s, diversity_reward
        
    def act(self, state, reward, learning=True):
        if self.experience_buffer.size() > self.batch_size and self.total_steps % self.update_freq == 0:
            s, a, r, s2, t, o = self.experience_buffer.sample_op(self.batch_size)
            self.train_batch(s, a, r, s2, t, o, batch_size=self.batch_size)
        
        if self.current_skill is None:
            self.current_skill = self.sample_skill()
        # if random() < self.epsilon:
        #     # print("random")
        #     return choice(range(self.num_actions))
        # else:
        best, pa_s_ = self.sess.run([self.bestA, self.pa_s], feed_dict={self.prev_states: [state], self.skill: [self.current_skill]})
        # action = np.random.choice(self.num_actions, 1, p=probA_S)
        # print('P(A|S) =', probA_S)
        # print('A =', action)
        # print('pa_s_[0] =', pa_s_[0])
        # print('argmax(pa_s[0]) =', np.argmax(pa_s[0]))
        # print('best =', best)
        pa_s = pa_s_[0]
        action = np.random.choice(range(self.num_actions), p=pa_s)
        # print('act=', action)

        if not (self.prev_state is None) and not (self.prev_action is None) and learning:
            self.experience_buffer.add([self.prev_state, self.prev_action, reward, state, state.is_terminal(), self.current_skill])


        self.prev_state, self.prev_action = state, action

        self.total_steps += 1

        return action

        
    def train_batch(self, prev_s, a, r, s, t, o, batch_size, external_r=False):

        # TODO: is the a better way to do this operation?
        pz_ = []
        for i in range(batch_size):
            pz_.append(self.pz[o[i]])
        pz = np.asarray(pz_)

        # pz = self.pz[o]
        # print('pz=', pz)
        skl = o
            
        diversity_reward = self.sess.run(self.diversity_reward, feed_dict={self.states: s, self.pz_tf: pz, self.skill: skl})
        V = self.sess.run(self.predictedVvalue, feed_dict={self.prev_states: prev_s, self.skill: skl})

        nextV = self.sess.run(self.predictedVvalue, feed_dict={self.prev_states: s, self.skill: skl})
        
        
        entropy = self.sess.run(self.entropy_out, feed_dict={self.prev_states: prev_s, self.actions: a, self.skill: skl})
        # print("entropy = ", entropy)

        # q = self.sess.run(self.predictedQvalue, feed_dict={self.prev_states: prev_s, self.actions: a})
        advantages = np.zeros_like(r)
        for i in range(len(advantages)):
            advantages[i] = diversity_reward[i] + nextV[i] - V[i] # gamma r
        # print("r=", r)

        # print("diversity_reward=", diversity_reward)
        # print("advantages=", advantages)
        # print("##################")


        # TODO: Can we refactor this into a call to actor and critic separately?

        _, self.al = self.sess.run([self.optimize_act, self.actor_loss], feed_dict={self.prev_states: prev_s, self.actions: a, self.skill: skl, self.advantages: advantages})

        _, self.cl = self.sess.run([self.optimize_crit, self.critic_loss], feed_dict={self.prev_states: prev_s, self.actions: a, self.skill: skl, self.states: s, self.entropies: entropy, self.diversity_reward: diversity_reward, self.pz_tf: pz})

        _, self.dl = self.sess.run([self.optimize_dsc, self.discriminator_loss], feed_dict={self.skill: skl, self.states: s})
        
        # _, self.l, self.al, self.cl, self.dl, summary = self.sess.run([self.optimize, self.loss, self.actor_loss, self.critic_loss, self.discrimator_loss, self.summary], feed_dict={self.prev_states: prev_s, self.actions: a, self.skill: skl, self.rewards: r, self.states: s, self.entropies: entropy, self.diversity_reward: diversity_reward, self.advantages: advantages, self.pz_tf: pz})
        
        # self.train_writer.add_summary(summary, self.num_train)
        # self.num_train += 1
        # print('loss, actor, critic, ent = %.2f, %.2f, %.2f, %.2f' % (l, al, cl, ent))
        # print('loss of actor, critic, discrim =', self.al, self.cl, self.dl)
        l = self.al + self.cl + self.dl

        
        return l

    def end_of_episode(self):
        self.current_skill = None

        self.prev_state, self.prev_action = None, None
        

    def reset(self):
        self.experience_buffer = ExperienceBuffer(buffer_size=self.buffer_size)
        self.prev_state, self.prev_action = None, None
        self.curr_step, self.total_steps = 0, 0
        self.curr_episode = 0

        # Initialize the network
        self.sess.run(self.initializer, feed_dict={})

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

    def log(self):
        print("actor, critic, discriminator loss=", self.al, self.cl, self.dl)
    
