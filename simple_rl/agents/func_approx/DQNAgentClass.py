''' DQNAgentClass.py: Class for Deep Q-network agent. Built based on the network
in DeepMind, Multi-agent RL in Sequential Social Dilemmas paper. '''

# Python imports.
import tensorflow as tf
import tflearn
import numpy as np
import random

# Other imports.
from simple_rl.agents.AgentClass import Agent
from simple_rl.agents.func_approx.ExperienceBuffer import ExperienceBuffer

class DQNAgent(Agent):

    NAME = "dqn"

    def __init__(self, sess=None, obs_dim=None, num_actions=0, buffer_size=100000, gamma=0.99, epsilon=0.05, learning_rate=0.001, tau=0.001, conv=False, name=NAME):
        Agent.__init__(self, name=name, actions=range(num_actions))

        if sess is None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
        else:
            self.sess = sess

        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.tau = tau

        self.update_freq = 1
        self.batch_size = 64

        self.conv = conv

        self.mainQ = QNetwork(sess=self.sess, learning_rate=self.learning_rate, obs_dim=self.obs_dim, num_actions=self.num_actions, conv=self.conv, name=name+"_main_q")
        self.targetQ = QNetwork(sess=self.sess, learning_rate=self.learning_rate, obs_dim=self.obs_dim, num_actions=self.num_actions, conv=self.conv, name=name+"_target_q")

        self.network_params = tf.trainable_variables(scope=self.name+"_main_q")
        self.target_network_params = tf.trainable_variables(scope=self.name+"_target_q")
        self.update_target_params = \
                                    [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                                          tf.multiply(self.target_network_params[i], 1.0 - self.tau))
                                     for i in range(len(self.target_network_params))]

        
        self.saver = tf.train.Saver(self.network_params + self.target_network_params)
        
        self.reset()

        # Load model from a checkpoint
        # if not (from_checkpoint is None):
        #     self.saver.restore(self.sess, from_checkpoint)
        #     print('Restored model from checkpoint: {}'.format(from_checkpoint))

    def act(self, state, reward, learning=True):
        '''
        Args:
            state (simple_rl.State)
            reward (float)

        Returns:
            (str)
        '''
        # Training
        if self.total_steps > 0 and self.total_steps % self.update_freq == 0 and self.experience_buffer.size() > self.batch_size and learning:
            s, a, r, s2, t = self.experience_buffer.sample(self.batch_size)
            self.train_batch(s, a, r, s2, t, batch_size=self.batch_size)

        state_d = state.data.flatten()
            
        # Not Training (or after training)
        if random.random() < self.epsilon:
            action =  np.random.choice(self.num_actions) # NOTE:  Again assumes actions encoded as integers
        else:
            action = self.mainQ.get_best_action(state_d)[0]

        if not (self.prev_state is None) and not (self.prev_action is None) and learning:
            self.experience_buffer.add((self.prev_state, self.prev_action, reward, state_d, state.is_terminal()))

        self.prev_state, self.prev_action = state_d, action

        # Saving checkpoints (NOTE: We only save checkpoints when training)
        if self.should_save and self.total_steps > 0 and self.total_steps % self.save_every == 0:
            save_path = self.saver.save(self.sess, '/tmp/{}.ckpt'.format(self.name))
            print('At step {}, saved model to {}'.format(self.total_steps, save_path))

        self.curr_step += 1
        self.total_steps += 1
        if state.is_terminal():
            self.curr_step = 0
            self.curr_episode += 1

        self.action_counts[action] += 1
        return action

    def get_q_value(self, s, a):
        return self.targetQ.get_q_value(s, a)

    def train_batch(self, s, a, r, s2, t, duration=None, batch_size=1):
        # print('s.shape=', s.shape)
        # print('a.shape=', a.shape)
        # print('s2.shape=', s2.shape)
        
        targetVals = self.targetQ.predict_value(s2)

        # Compute y-vals
        y = np.zeros(batch_size)
        for i in range(batch_size):
            if t[i]:
                y[i] = r[i]
            else:
                y[i] = r[i] + targetVals[i]
        _, l = self.mainQ.train(s, a, np.reshape(y, (batch_size, 1)))

        if self.print_loss and (self.total_steps % self.print_every == 0) and self.total_steps > 0:
            print('Loss for step {}: {}'.format(self.total_steps, l))

        self.sess.run(self.update_target_params, feed_dict={})

    def __str__(self):
        return str(self.name)

    def reset(self):
        self.mainQ.initialize()
        self.targetQ.initialize()
        
        self.experience_buffer = ExperienceBuffer(buffer_size=self.buffer_size)
        
        self.prev_state, self.prev_action = None, None
        self.curr_step, self.total_steps = 0, 0
        self.curr_episode = 0
        self.total_reward = 0
        # self.curr_instances += 1

        self.action_counts = [0] * self.num_actions
        
        self.should_save, self.save_every = True, 100000
        self.print_loss, self.print_every = True, 10000

    def restore(self, directory, rev=False, name=None):
        if name is None:
            name = self.name

        if rev: 
            self.saver.restore(self.sess, directory + '/' + self.name + 'rev')
        else:
            self.saver.restore(self.sess, directory + '/' + self.name)
            
    def save(self, directory, name=None):
        if name is None:
            self.saver.save(self.sess, directory + '/' + self.name)
        else:
            self.saver.save(self.sess, directory + '/' + name)


# --------------
# -- QNetwork --
# --------------
class QNetwork():
    def __init__(self, sess, learning_rate=1e-4, obs_dim=None, num_actions=None, conv=False, name=None):
        self.sess = sess
        self.learning_rate = learning_rate
        self.num_actions = num_actions
        self.obs_dim = obs_dim
        self.conv = conv
        self.name = name
        self.obs, self.q_estm = self.q_network(scope=name + "_q")
                
        self.best_action = tf.argmax(self.q_estm, 1)

        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1], name=name+"_pred_q")

        self.actions = tf.placeholder(tf.int32, [None], name=name+"_actions")
        actions_onehot = tf.one_hot(self.actions, self.num_actions, dtype=tf.float32)
        Q = tf.reduce_sum(actions_onehot * self.q_estm, axis=1)
        # TODO: add entropy loss
        self.loss = tflearn.mean_square(self.predicted_q_value, Q)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.optimize = self.optimizer.minimize(self.loss)
        
        self.network_params = tf.get_collection(tf.GraphKeys.VARIABLES, scope=name + "_q")
        self.initializer = tf.initializers.variables(self.network_params + self.optimizer.variables())        

    def q_network(self, scope):
        obs = tf.placeholder(tf.float32, shape=[None, self.obs_dim], name=scope+"_obs")
        if self.conv:
            # obs = tf.placeholder(tf.float32, shape=[None, 105, 80, 3], name=scope+'_obs')
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                reshaped_obs = tf.reshape(obs, [-1, 105, 80, 3])
                net = tflearn.conv_2d(reshaped_obs, 32, 8, strides=4, activation='relu')
                net = tflearn.conv_2d(net, 64, 4, strides=2, activation='relu')
                out = tflearn.fully_connected(net, self.num_actions, weights_init=tflearn.initializations.uniform(minval=-0.003, maxval=0.003))
        else:    
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                net = tflearn.fully_connected(obs, 400, name='d1', weights_init=tflearn.initializations.truncated_normal(stddev=1.0/float(self.obs_dim)))
                net = tf.layers.batch_normalization(net)
                net = tflearn.activations.relu(net)
                # net = tflearn.fully_connected(net, 300, name='d2', weights_init=tflearn.initializations.truncated_normal(stddev=1.0/400.0)) # This gives the values from the observation
                # net = tflearn.activations.relu(net)

                w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
                out = tflearn.fully_connected(net, self.num_actions, weights_init=w_init)

        return obs, out

    def train(self, obs, actions, predicted_q_value):
        
        return self.sess.run([self.optimize, self.loss], feed_dict={
            self.obs: obs,
            self.actions: actions,
            self.predicted_q_value: predicted_q_value
        })

    def get_q_value(self, state, action):
        vals = self.sess.run(self.q_estm, feed_dict={
            self.obs: [state]
        })
        return vals[0][action]

    def predict_value(self, state):
        vals = self.sess.run(self.q_estm, feed_dict={
            self.obs: state
        })
        return np.max(vals, axis=1)

    def get_best_action(self, obs):
        # print('obs_dim=', self.obs_dim)
        # print('obs=', obs.shape)
        obs_ = np.reshape(obs, (1, self.obs_dim))
        return self.sess.run(self.best_action, feed_dict={
            self.obs: obs_
        })

    def initialize(self):
        return self.sess.run(self.initializer, feed_dict={})
