import tensorflow as tf
import numpy as np

def leakyReLU(x, alpha=0.05):
    return tf.maximum(x, x * alpha)

def logClp(x, lower=0.000001, upper=0.999999):
    return tf.log(tf.clip_by_value(x, lower, upper))

def logClpNp(x, lower=0.000001, upper=0.999999):
    return np.log(np.clip(x, lower, upper))

def one_hot(x, size, t=1.0, f=0.0):
    # print('type(x)', type(x))
    # assert(isinstance(x, int))
    ret = np.full(size, f)
    ret[x] = t
    return ret
