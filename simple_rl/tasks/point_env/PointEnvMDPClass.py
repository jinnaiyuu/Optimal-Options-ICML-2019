# Python imports.
import numpy as np
import os

# Other imports.
from mujoco_py import load_model_from_path, MjSim, MjViewer, MjSimState
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.point_env.PointEnvStateClass import PointEnvState


class PointEnvMDP(MDP):
    def __init__(self, torque_multiplier=50., init_mean=(-0.2, -0.2), render=False):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(dir_path, 'asset/point_mass.xml')
        model = load_model_from_path(path)
        self.sim = MjSim(model)
        self.render = render
        self.init_mean = init_mean

        self.viewer = MjViewer(self.sim)

        # Config
        self.env_name = "Point-Mass-Environment"
        self.target_position = np.array([0., 0.])
        self.target_tolerance = 0.01
        self.init_noise = 0.05
        self.max_absolute_torque = 5.
        self.torque_multiplier = torque_multiplier

        self._initialize_mujoco_state()
        self.init_state = self.get_state()

        MDP.__init__(self, [0, 1], self._transition_func, self._reward_func, self.init_state)

    def _reward_func(self, state, action):
        self.next_state = self._step(action)
        if self.render: self.viewer.render()
        reward = +0 if self.next_state.is_terminal() else -1.
        return reward

    def _transition_func(self, state, action):
        return self.next_state

    def execute_agent_action(self, action, option_idx=None):
        reward, next_state = super(PointEnvMDP, self).execute_agent_action(action)
        return reward, next_state

    def is_goal_position(self, position):
        distance = np.linalg.norm(position - self.target_position)
        return distance <= self.target_tolerance

    def get_state(self):
        position = self.sim.data.qpos
        velocity = self.sim.data.qvel
        done = self.is_goal_position(position)
        state = PointEnvState(position, velocity, done)
        return state

    def _step(self, action):
        self.sim.data.ctrl[:] = self.torque_multiplier * action
        self.sim.step()
        return self.get_state()

    def _initialize_mujoco_state(self):
        init_position = np.array(self.init_mean) + np.random.uniform(0., self.init_noise, 2)
        init_state = MjSimState(time=0., qpos=init_position, qvel=np.array([0., 0.]), act=None, udd_state={})
        self.sim.set_state(init_state)

    def reset(self):
        self._initialize_mujoco_state()
        self.init_state = self.get_state()

        super(PointEnvMDP, self).reset()

    def __str__(self):
        return self.env_name
