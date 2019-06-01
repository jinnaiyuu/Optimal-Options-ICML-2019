# from sklearn.svm import SVC
# from sklearn.linear_model import SGDClassifier

import numpy as np

class ClassifierPredicate(object):

	def __init__(self, true_data, method='SVC', true_if_in=True):
                
                x = np.stack(true_data).reshape((len(true_data), length))
                y = np.ones(len(true_data), dtype=int)

                n_falses = len(true_data) * 10
                x_ = np.random.randint(0, 7, n_falses * 210 * 160 * 3).reshape((n_falses, 210 * 160 * 3))
                y_ = np.zeros(n_falses, dtype=int)

                x__ = np.concatenate([x, x_])
                y__ = np.concatenate([y, y_])
                
                # print('xshape=', x__.shape)
                # print('yshape=', y__.shape)
                
                self.clf = SGDClassifier()
                self.clf.fit(x__, y__)

                self.method = method

                self.true_if_in = true_if_in

	def is_true(self, x):
                # Return the result
                x_ = x.data.flatten()
                # print('x_shape=', x_.shape)
                ret = self.clf.predict([x_])
                # print('ret=', ret[0])
                if ret[0] == 1:
                        return self.true_if_in
                else:
                        return not self.true_if_in
