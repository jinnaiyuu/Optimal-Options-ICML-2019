# Python imports
import numpy as np

# Local imports
from simple_rl.mdp.StateClass import State

''' GymStateClass.py: Contains a State class for Gym. '''

class GymState(State):
    ''' Gym State class '''

    def __init__(self, data=[], is_terminal=False):
        self.data = data
        State.__init__(self, data=data, is_terminal=is_terminal)

    def to_rgb(self, x_dim, y_dim):
        # 3 by x_length by y_length array with values 0 (0) --> 1 (255)
        board = np.zeros(shape=[3, x_dim, y_dim])
        # print self.data, self.data.shape, x_dim, y_dim
        return self.data

    # def getXY(self):
    #     # TODO: it has to be MontezumaRevenge-ram-v0.
    #     def getByte(ram, row, col):
    #         row = int(row, 16) - 8
    #         col = int(col, 16)
    #         return ram[row*16+col]
    #     x = getByte(self.data, 'a', 'a')
    #     y = getByte(self.data, 'a', 'b')
    #     x_img = int(160.0 * (float(x) - 1) / float((9 * 16 + 8) - 1))
    #     y_img = int(160.0 * (float(y) - (8 * 16 + 6)) / float((15 * 16 + 15) - (8 * 16 + 6)))
    # 
    #     return x_img, y_img

    def __eq__(self, other):
        return np.array_equal(self.data, other.data)

    def __hash__(self):
        return hash(self.data.tostring())
