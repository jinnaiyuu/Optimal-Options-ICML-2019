''' DDPGAgent. '''

# Python imports.
import tensorflow as tf
import numpy as np
import random
import tflearn


class Actor(object):
    def __init__(self, sess, obs_dim, action_dim, action_bound, learning_rate, tau, batch_size, lowmethod='ddpg', name='ddpg-actor'):
        assert(sess is not None)
        assert(obs_dim is not None)
        assert(action_dim is not None)
        assert(action_bound is not None)
        assert(type(learning_rate) is float)
        assert(type(tau) is float)
        assert(type(batch_size) is int)
        assert(batch_size > 0)

        assert(action_bound[0].shape == action_bound[1].shape)
        # assert(action_bound[0][0] == -action_bound[1][0])
        
        self.sess = sess
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        self.lowmethod = lowmethod
        self.name = name

        # prob_action is for discrete action space
        self.obs, self.action, self.scaled_action, self.prob_action = self.actor_network(scope=name + "_actor")
        
        self.network_params = tf.trainable_variables(scope=name + "_actor")

        self.target_obs, self.target_action, self.target_scaled_action, _ = self.actor_network(scope=name + "_actor_target")

        self.target_network_params = tf.trainable_variables(scope=name + "_actor_target")

        self.update_target_params = \
                                     [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                                           tf.multiply(self.target_network_params[i], 1.0 - self.tau))
                                      for i in range(len(self.target_network_params))]

        self.action_gradient = tf.placeholder(tf.float32, [None, self.action_dim], name='action_gradient')

        self.unnormalized_actor_gradients = tf.gradients(self.scaled_action, self.network_params, - self.action_gradient)

        # TODO: ValueError: None values not supported
        #       I have no idea where it comes from.
        
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        self.optimizer= tf.train.AdamOptimizer(self.learning_rate)
        self.optimize = self.optimizer.apply_gradients(zip(self.actor_gradients, self.network_params))

        p = tf.get_collection(tf.GraphKeys.VARIABLES, scope=name + "_actor")
        p_target = tf.get_collection(tf.GraphKeys.VARIABLES, scope=name + "_actor_target")

        self.initializer = tf.initializers.variables(p + p_target + self.optimizer.variables())

        self.saver = tf.train.Saver(self.network_params + self.target_network_params)
        
    def actor_network(self, scope):
        obs = tflearn.input_data(shape=[None, self.obs_dim], name='obs')

        if self.lowmethod == 'linear':
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                action = tflearn.fully_connected(obs, self.action_dim, weights_init=tflearn.initializations.truncated_normal(stddev=1.0/float(self.obs_dim)))
        else:
                
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                net = tflearn.fully_connected(obs, 400, weights_init=tflearn.initializations.truncated_normal(stddev=1.0/float(self.obs_dim)))
                net = tf.layers.batch_normalization(net)
                net = tflearn.activations.relu(net)
                net = tflearn.fully_connected(net, 300, weights_init=tflearn.initializations.truncated_normal(stddev=1.0/400.0))
                net = tf.layers.batch_normalization(net)
                net = tflearn.activations.relu(net)
                w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
                action = tflearn.fully_connected(net, self.action_dim, activation='tanh', weights_init=w_init)

        scaled_action = tf.multiply(action, self.action_bound[1])
        
        sigmd = tf.nn.sigmoid(action)
        pa_s = tf.nn.softmax(sigmd)
        return obs, action, scaled_action, pa_s

    # APIs
    def train(self, obs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.obs: obs,
            self.action_gradient: a_gradient
        })


    def predict(self, obs):
        return self.sess.run(self.scaled_action, feed_dict={
            self.obs: obs
        })

    def predict_target(self, obs):
        # print('obs=', obs) # List of arrays.
        # print('Actor::predict_target: obs.shape=', obs.shape)
        return self.sess.run(self.target_scaled_action, feed_dict={
            self.target_obs: obs
        })

    def predict_discrete(self, obs):
        return self.sess.run(self.prob_action, feed_dict={
            self.obs: obs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_params)
        
    # def get_num_trainable_vars(self):
    #    return self.num_trainable_vars

    def initialize(self):
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

        

# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class ActorNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        # TODO: Sigma is set to 1.0 in HERO paper.
        # mu   : Mean value.
        # sigma: ?
        # theta: ?
        # dt   : ?
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)
        
