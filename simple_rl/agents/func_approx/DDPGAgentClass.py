''' DDPGAgent. '''

# Python imports.
import tensorflow as tf
import numpy as np
import random
import tflearn

# Other imports.
from simple_rl.agents.AgentClass import Agent
from simple_rl.agents.func_approx.ExperienceBuffer import ExperienceBuffer
from simple_rl.agents.func_approx.ActorNetwork import Actor
from simple_rl.agents.func_approx.ActorNetwork import ActorNoise
from simple_rl.agents.func_approx.CriticNetwork import Critic
from simple_rl.mdp.StateClass import State

class DDPGAgent(Agent):

    NAME = "ddpg"

    def __init__(self, sess=None, obs_dim=None, action_dim=None, action_bound=None, buffer_size=100000, batch_size=64, name=NAME, actor_rate=0.0001, critic_rate=0.001, tau=0.001, should_train=True, from_checkpoint=None, gamma=0.99):
        # TODO: Use a shared experience buffer?
        
        Agent.__init__(self, name=name, actions=[])

        assert(type(obs_dim) is int)
        assert(type(action_dim) is int)
        assert(action_bound is not None)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.buffer_size = buffer_size

        self.gamma = gamma
        
        self.should_train = should_train

        # Fixed parameters
        self.update_freq = 1 # 64
        self.batch_size = batch_size
        self.should_save, self.save_every = True, 100000
        self.print_loss, self.print_every = True, 10000

        if sess is None:
            self.sess = tf.Session()
        else:
            self.sess = sess
        self.actor = Actor(sess=self.sess, obs_dim=self.obs_dim, action_dim=self.action_dim, \
                           action_bound=self.action_bound, learning_rate=actor_rate, tau=tau, batch_size=self.batch_size, name=name)

        self.critic = Critic(sess=self.sess, obs_dim=self.obs_dim, action_dim=self.action_dim, \
                             learning_rate=critic_rate, tau=tau, name=name)

        self.actor_noise = ActorNoise(mu=np.zeros(self.action_dim), sigma=0.3)


        self.total_reward = 0
        
        self.reset()

        # Load model from a checkpoint
        # if not (from_checkpoint is None):
        #     self.saver.restore(self.sess, from_checkpoint)
        #     print('Restored model from checkpoint: {}'.format(from_checkpoint))

    def act(self, state, reward, learning=True, add_noise=True):
        '''
        Args:
            state (simple_rl.State)
            reward (float)

        Returns:
            (str)
        '''
        # Training
        if self.should_train and self.total_steps > 0 and self.total_steps % self.update_freq == 0 and self.experience_buffer.size() > self.batch_size and learning:
            s, a, r, s2, t = self.experience_buffer.sample(self.batch_size)
            self.train_batch(s, a, r, s2, t, batch_size=self.batch_size)

        # TODO: Convert a state to a numpy array so that we can send to the Tensorflow.
        # img = state.data # TODO: does it work?
        # print('state.data=', state.data)
        # print('type(state.data)=', type(state.data))

        # action = self.actor.predict(img) + self.actor_noise()
        action_ = self.actor.predict(np.reshape(state.data, (1, self.obs_dim)))

        # TODO: Generate a standard deviation from a neural network.
        
        if add_noise:
            # TODO: Scale to the size of the 
            noise = self.actor_noise()
            # scaled_noise = noise * self.action_bound[1]

            action_ = action_ + noise
            
        action = action_[0]

        if not (self.prev_state is None) and not (self.prev_action is None) and learning:
            self.experience_buffer.add((self.prev_state, self.prev_action, reward, state, state.is_terminal()))
            # self.experience_buffer.add((np.reshape(self.prev_state, (self.obs_dim,)), np.reshape(self.prev_action, (self.action_dim,)), reward, np.reshape(img, (self.obs_dim,)), state.is_terminal()))

        self.prev_state, self.prev_action = state, action

        # Saving checkpoints (NOTE: We only save checkpoints when training)
        # if self.should_train and self.should_save and self.total_steps > 0 and self.total_steps % self.save_every == 0:
        #     save_path = self.saver.save(self.sess, '/tmp/{}.ckpt'.format(self.name))
        #     print('At step {}, saved model to {}'.format(self.total_steps, save_path))

        self.curr_step += 1
        self.total_steps += 1

        self.total_reward += reward

        # TODO: Check why this is_terminal is always called on the first step of the episode.
        # if state.is_terminal() and self.curr_step > 1:
            # print('terminal state =', state.data)
        #     print('#Episode=', self.curr_episode, '#steps=', self.curr_step, 'Total_reward=', self.total_reward)
        #     self.curr_step = 0
        #     self.curr_episode += 1
        #     self.total_reward = 0
        # print('action selected =', action)

        return action

    def train_batch(self, s, a, r, s2, t, duration=None, batch_size=1):
        # TODO: Should you update the actor using off-policy trajectories?
        
        s_tnsr = self.convert_states_to_arrays(s)
        s2_tnsr = self.convert_states_to_arrays(s2)
        
        # TODO: we do not support a case where duration is not 1.
        assert(duration is None)
        
        if duration is None:
            duration = [1] * len(s)

        # TODO: We are getting 
        target_q = self.critic.predict_target(s2_tnsr, self.actor.predict_target(s2_tnsr))
        
        # Compute y-vals
        y = []
        for i in range(batch_size):
            if t[i]:
                # y.append(np.array([r[i]]))
                y.append(r[i])
            else:
                # y.append(np.array([r[i] + self.gamma * target_q[i]]))
                y.append(r[i] + self.gamma * float(target_q[i]))

        predicted_q_value, _ = self.critic.train(s_tnsr, a, np.reshape(y, (batch_size, 1)))

        a_outs = self.actor.predict(s_tnsr)
        grads = self.critic.action_gradients(s_tnsr, a_outs)
        self.actor.train(s_tnsr, grads[0])
        
        self.actor.update_target_network()
        self.critic.update_target_network()

    def end_of_episode(self):
        print('#Episode=', self.episode_number, '#steps=', self.curr_step, 'Total_reward=', self.total_reward)
        self.curr_step = 0
        Agent.end_of_episode(self)
        
    def reset(self):
        
        self.experience_buffer = ExperienceBuffer(buffer_size=self.buffer_size)
        self.prev_state, self.prev_action = None, None
        self.curr_step, self.total_steps = 0, 0
        self.curr_episode = 0
        # self.saver = tf.train.Saver()

        # TODO: I believe global variable initializer gonna break everything.
        #       so we have to instead just initialize the network of the agent.
        # self.sess.run(tf.global_variables_initializer())

        self.critic.initialize()
        self.actor.initialize()

        self.critic.update_target_network()
        self.actor.update_target_network()
        
    def convert_states_to_arrays(self, obs):
        assert(isinstance(obs, list))

        arrays = []
        for s in obs:
            assert(isinstance(s, State))
            arrays.append(s.data)

        # tnsr = np.stack(arrays, axis=0)
        return arrays

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
        
    def __str__(self):
        return str(self.name)

