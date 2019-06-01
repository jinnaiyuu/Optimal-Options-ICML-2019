import os
import numpy as np
import pickle
from simple_rl.tasks import GymState

# -----------------------
# -- Experience Buffer --
# -----------------------
class ExperienceBuffer():
    # TODO: The implementation is inefficient in terms of memory and time.

    def __init__(self, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        while len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)

        self.buffer.append(experience)

    def sample(self, size):
        indexes = np.random.randint(0, high=len(self.buffer), size=size)
        s1 = [self.buffer[index][0] for index in indexes]
        a = [self.buffer[index][1] for index in indexes]
        r = [self.buffer[index][2] for index in indexes]
        s2 = [self.buffer[index][3] for index in indexes]
        t = [self.buffer[index][4] for index in indexes]
        return [s1, a, r, s2, t]

    def sample_op(self, size):
        indexes = np.random.randint(0, high=len(self.buffer), size=size)
        s1 = [self.buffer[index][0] for index in indexes]
        a = [self.buffer[index][1] for index in indexes]
        r = [self.buffer[index][2] for index in indexes]
        s2 = [self.buffer[index][3] for index in indexes]
        t = [self.buffer[index][4] for index in indexes]
        stps = [self.buffer[index][5] for index in indexes]
        return [s1, a, r, s2, t, stps]

    def size(self):
        return len(self.buffer)

    def restore(self, filename):
        infile = open(filename, 'rb')
        # self.buffer = pickle.load(infile, protocol=0)
        self.buffer = pickle.load(infile)
        infile.close()

    def save(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        outfile = open(filename, 'wb')
        # TODO: pickle is not the most efficient way to store the data.
        #       Because the state can be any type, it is hard to optimize.
        # pickle.dump(self.buffer, outfile, protocol=0)
        pickle.dump(self.buffer, outfile, protocol=-1)
        outfile.close()


    def save_sao(self, filename):
        self.save_sa(filename)

        o = [self.buffer[index][5] for index in range(len(self.buffer))]
        oarr = np.asarray(o)
        np.save(filename + 'option', oarr)

    def save_sa(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        s = [self.buffer[index][0].data for index in range(len(self.buffer))]
        a = [self.buffer[index][1] for index in range(len(self.buffer))]

        # pos = np.argwhere(s[-10] > 0)
        # print('nonzerso=', pos)
        
        sarr = np.asarray(s)
        aarr = np.asarray(a)

        np.save(filename + 'state', sarr)
        np.save(filename + 'action', aarr)


    def restore_sao(self, filename):
        # TODO: How should we store the (s, a) pairs?
        sarr = np.load(filename + 'state.npy')
        aarr = np.load(filename + 'action.npy')
        oarr = np.load(filename + 'option.npy')

        size = sarr.shape[0]

        for i in range(0, size-1):
            s1 = GymState(sarr[i])
            a = aarr[i]
            r = 0
            s2 = GymState(sarr[i+1])
            t = False
            o = oarr[i]
            exp = (s1, a, r, s2, t, o)
            self.add(exp)
        
    def restore_sa(self, filename):
        # TODO: How should we store the (s, a) pairs?
        sarr = np.load(filename + 'state.npy')
        aarr = np.load(filename + 'action.npy')

        size = sarr.shape[0]

        for i in range(0, size-1):
            s1 = GymState(sarr[i])
            a = aarr[i]
            r = 0
            s2 = GymState(sarr[i+1])
            t = False
            exp = (s1, a, r, s2, t)
            self.add(exp)
        
        # s, a, r, s2, t = self.sample(1)
        # print('a.shape=', a.shape)
        
# Used to update TF networks
# def update_target_graph(tfVars,tau):
#     total_vars = len(tfVars)
#     op_holder = []
#     for idx,var in enumerate(tfVars[0:total_vars//2]):
#         op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
#     return op_holder
# 
# def update_target(op_holder,sess):
#     for op in op_holder:
#         sess.run(op)
# 
