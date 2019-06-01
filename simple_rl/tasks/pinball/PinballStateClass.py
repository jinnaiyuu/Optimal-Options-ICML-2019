# Python imports.
from __future__ import print_function
import numpy as np

# Other imports.
from simple_rl.mdp.StateClass import State

class PinballState(State):
    def __init__(self, x, y, xdot, ydot, is_terminal=False):
        self.x = x
        self.y = y
        self.xdot = xdot
        self.ydot = ydot

        data = np.asarray([x, y, xdot, ydot])

        State.__init__(self, data=data, is_terminal=is_terminal)

    def get_position(self):
        return np.array([self.x, self.y])

    def convert_to_positional_state(self):
        return PositionalPinballState(self.x, self.y, self.is_terminal())

    def state_space_size(self):
        return self.data.shape[0]

    def __hash__(self):
        return hash(str(self.data))

    def __str__(self):
        return "(x: {}, y: {}, xdot: {}, ydot: {}, term: {})".format(self.data[0], self.data[1], self.data[2], self.data[3], self.is_terminal())

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return isinstance(other, PinballState) and self.data == other.data

    def __ne__(self, other):
        return not self == other
