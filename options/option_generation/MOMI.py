#!/usr/bin/env python
# Libraries
import numpy as np

from options.graph.set_cover import SC_APPROX, SC_APPROX2, SC_OPT
from options.util import DeriveGraph
    
def MOMI(mdp, distance, l, solver):

    X = DeriveGraph(distance, l - 1) + np.identity(distance.shape[0])

    # Remove states which is already reachable within l steps
    xg = []
    for s in range(X.shape[0]):
        if all(X[s] <= l):
            xg.append(s)
    
    if solver == 'chvatal':
        print("MOMI(l =", l, ", chvatal)")
        C = SC_APPROX2(X.transpose())
    elif solver == 'hochbaum':
        print("MOMI(l =", l, ", hochbaum)")
        C = SC_APPROX(X)
    elif solver == 'optimal':
        print("MOMI(l =", l, ", OPT)")
        C = SC_OPT(X.transpose())
    else:
        print('unknown solver for set cover', approx)
        assert(False)
        exit(0)

    return C
    
if __name__ == "__main__":
    pass
