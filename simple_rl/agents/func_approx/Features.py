'''
LinearQLearningAgentClass.py

Contains implementation for a Q Learner with a Linear Function Approximator.
'''

# Python imports.
import numpy as np
from numpy.linalg import norm
import math

# Other imports.
from collections import defaultdict


class Feature(object):
    def __init__(self, state_dim, num_actions):
        # TODO: Assuming action set is discrete.
        self.state_dim = state_dim
        self.num_actions = num_actions

    def feature(self, state, action):
        # Return a list of feature values.
        f = [0] * self.state_dim
        return f

    def alpha(self):
        return [1] * self.state_dim
    
    def num_features(self):
        return self.state_dim
        
class RBF(object):
    def __init__(self, state_dim, num_actions):
        self.state_dim = state_dim

    def feature(self, state, action):
        fs = [0] * self.state_dim

        for i, f in enumerate(fs):
            fs[i] = math.exp(-(f)**2)
        return fs
    
    def num_features(self):
        return self.state_dim
    
class Fourier(object):
    def __init__(self, state_dim, bound, order):
        assert(type(state_dim) is int)
        assert(type(order) is int)

        assert(state_dim == bound[0].shape[0])
        assert(state_dim == bound[1].shape[0])
        
        self.state_dim = state_dim
        self.state_up_bound = bound[0]
        self.state_low_bound = bound[1]
        self.order = order


        self.coeff = np.indices((self.order,) * self.state_dim).reshape((self.state_dim, -1)).T

        n = np.array(list(map(norm, self.coeff)))
        n[0] = 1.0
        self.norm = 1.0 / n
        
    def feature(self, state, action):
        xf = state.data.flatten()
        assert(xf.shape[0] == self.state_dim)

        norm_state = (xf + self.state_low_bound) / (self.state_up_bound - self.state_low_bound)
        
        f_np = np.cos(np.pi * np.dot(self.coeff, norm_state))

        # Check if the weights are set to numbers
        assert(not np.isnan(np.sum(f_np)))

        return f_np.tolist()

    def alpha(self):
        return self.norm

    def num_features(self):
        # What is the number of features for Fourier?
        return self.order**(self.state_dim)

class Subset(object):
    def __init__(self, state_dim, feature_indices):
        self.state_dim = state_dim
        self.indices = feature_indices
        self.nf = len(feature_indices)

    def feature(self, state, action):
        ret = []
        for i in self.indices:
            ret.append(state.data[i])
        return ret

    def num_features(self):
        return self.nf

    def alpha(self):
        return [1.0] * self.nf

class Monte(object):
    def __init__(self):
        pass

    def feature(self, state, action):
        x = self.getByte(state.data, 'a', 'a')
        y = self.getByte(state.data, 'a', 'b')
        x_img = 160.0 * (float(x) - 1) / float((9 * 16 + 8) - 1)
        y_img = 160.0 * (float(y) - (8 * 16 + 6)) / float((15 * 16 + 15) - (8 * 16 + 6))

        return [x_img, y_img]

    def alpha(self):
        return [1.0, 1.0]

    def num_features(self):
        return 2

    def getByte(self, ram, row, col):
        row = int(row, 16) - 8
        col = int(col, 16)
        return ram[row*16+col]


class AgentPos(object):
    def __init__(self, game):
        self.game = game
        
        if 'Freeway' in self.game:
            self.target = (252, 252, 84)
        elif 'MsPacman' in self.game:
            self.target = (210, 164, 74)
        else:
            assert(False)


    def feature(self, state, action):
        img = np.asarray(state.data)
        # print('img.shape=', img.shape)
        img = np.reshape(img, (210, 160, 3))
        # img = np.reshape(img, (105, 80, 3))
        shape = img.shape
        pos = None
        # print('img=', img)
        for y in range(shape[0]):
            for x in range(shape[1]):
                # print(img[y][x])
                if tuple(img[y][x]) == self.target:
                    # TODO: The color of the agent is the same as that of the background color. We should fix it before the final figure.
                    if self.game == "Freeway-v0" and 75 < y and y < 125:
                        continue
                    pos = (x + 2, 210 - y)
                    return [x + 2, 210 - y]
        print('no agent found')
        return [0, 0]

    def alpha(self):
        return [1.0, 1.0]

    def num_features(self):
        return 2
