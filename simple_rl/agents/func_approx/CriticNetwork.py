''' DDPGAgent. '''

# Python imports.
import tensorflow as tf
import numpy as np
import random
import tflearn


class Critic(object):
    def __init__(self, sess, obs_dim, action_dim, tau, learning_rate, lowmethod='ddpg', name='ddpg-critic'):
        # TODO: num_actor_vars?
        self.sess = sess
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.lowmethod = lowmethod
        self.name = name

        self.obs, self.action, self.q_estm = self.critic_network(scope=name + "_critic")

        # TODO: What does trainable_variables retrieve?
        self.network_params = tf.trainable_variables(scope=name + "_critic")

        # print('Critic network_params=', self.network_params)

        self.target_obs, self.target_action, self.target_q_estm = self.critic_network(scope=name + "_critic_target")

        self.target_network_params = tf.trainable_variables(scope=name + "_critic_target")

        # Calling this would update the parameters for the target network.
        self.update_target_params = \
                                     [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
                                     + tf.multiply(self.target_network_params[i], 1.0 - self.tau)) \
                                     for i in range(len(self.target_network_params))]

        # y_i
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1], name='predicted_q_value')
        
        # TODO: Implement weight decay
        self.loss = tflearn.mean_square(self.predicted_q_value, self.q_estm) # + tf.reduce_sum(self.network_params ** 2)

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.optimize = self.optimizer.minimize(self.loss)

        self.action_grads = tf.gradients(self.q_estm, self.action)

        
        p = tf.get_collection(tf.GraphKeys.VARIABLES, scope=name + "_critic")
        p_target = tf.get_collection(tf.GraphKeys.VARIABLES, scope=name + "_critic_target")

        self.initializer = tf.initializers.variables(p + p_target + self.optimizer.variables())

        self.saver = tf.train.Saver(self.network_params + self.target_network_params)

    def critic_network(self, scope):
        obs = tflearn.input_data(shape=[None, self.obs_dim], name='obs')
        action = tflearn.input_data(shape=[None, self.action_dim], name='action')
        if self.lowmethod == 'linear':
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                out = tflearn.fully_connected(tf.concat([obs, action], 1), 1, tflearn.initializations.uniform(minval=-0.003, maxval=0.003))
                
        else:
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                net = tflearn.fully_connected(obs, 400, name='d1', weights_init=tflearn.initializations.truncated_normal(stddev=1.0/float(self.obs_dim)))
                net = tf.layers.batch_normalization(net)
                net = tflearn.activations.relu(net)

                t1 = tflearn.fully_connected(net, 300, name='d2', weights_init=tflearn.initializations.truncated_normal(stddev=1.0/400.0)) # This gives the values from the observation
                t2 = tflearn.fully_connected(action, 300, name='d3', weights_init=tflearn.initializations.truncated_normal(stddev=1.0/self.action_dim)) # This gives the values from the action

                net = tflearn.activation(tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

                w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
                out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return obs, action, out

    
    # Methods below are the API for the network.     
    def train(self, obs, action, predicted_q_value):
        # TODO: Convert the MDPState to observations.
        # print('obs=', obs)
        # print('action=', action)
        # print('predicted_q_value=', predicted_q_value)
        return self.sess.run([self.q_estm, self.optimize], feed_dict={
            self.obs: obs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, obs, action):
        return self.sess.run(self.q_estm, feed_dict={
            self.obs: obs,
            self.action: action
        })

    def predict_target(self, obs, action):
        return self.sess.run(self.target_q_estm, feed_dict={
            self.target_obs: obs,
            self.target_action: action
        })

    def action_gradients(self, obs, action):
        return self.sess.run(self.action_grads, feed_dict={
            self.obs: obs,
            self.action: action
        })

    def update_target_network(self):
        self.sess.run(self.update_target_params)

    def initialize(self):
        # TODO: returning None?
        return self.sess.run(self.initializer, feed_dict={})
        
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
