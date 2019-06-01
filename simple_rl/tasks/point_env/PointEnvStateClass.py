# Python imports.
import numpy as np

# Other imports.
from simple_rl.mdp.StateClass import State


class PointEnvState(State):
    def __init__(self, position, velocity, done):
        self.position = position
        self.velocity = velocity

        State.__init__(self, np.concatenate((position, velocity), axis=0), is_terminal=done)

