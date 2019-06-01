#!/usr/bin/env python

import numpy as np

from options.graph.asymmetric_k_center import GetRadius, AKC_APPROX, AKC_OPT

def MIMO(mdp, distance, k, solver):
    if solver == 'archer':
        print("MIMO(k =", k, ", archer)")
        C, Rbound = AKC_APPROX(distance, k)
        R = GetRadius(distance.transpose(), C)
    elif solver == 'optimal':
        print("MIMO(k =", k, ", optimal)")
        C, R = AKC_OPT(distance, k)
    else:
        print('Unknown solver for asymk', solver)
        assert(False)

    # backups = (R+1) * (len(sIndex) * len(mdp.actions) + k)
    # print("distance=")
    # print(distance)
    print("centers=", C)

    return C, R

if __name__ == "__main__":
    pass
