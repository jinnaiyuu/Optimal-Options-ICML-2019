import math
import numpy as np
import copy
from collections import defaultdict
from options.util import neighbor

def RestrictedShortestPath(graph, c, d, D, eps, v_init, v_goal):
    # TODO: Hassin'90 (approx. algorithm)

    ############################################
    # TODO: The implementation is pseudo-polinomial exact algorithm for now.

    N = int(graph.shape[0]) # N: number of states

    Fs = []
    F = defaultdict(lambda: 1000000)
    Fs.append(F)
    
    for cur_d in range(1, D + 1):
        F[v_init] = 0

        for j in range(N):
            if j == v_init:
                continue

            F[j] = Fs[cur_d - 1][j]
            
            nbor = neighbor(graph, j)
            for k in nbor:
                dcost = d[k][j]
                ccost = c[k][j]
                if cur_d - dcost >= 0:
                    # print('cur_d', cur_d)
                    # print('dcost', dcost)
                    # print('j, k=', j, ', ', k)
                    newCost = Fs[cur_d - dcost][k] + ccost
                if newCost < F[j]:
                    F[j] = newCost

        Fs.append(F)

        # print('F=', F)
    return Fs[D - 1][v_goal]

def Tests():
    G = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]], dtype=int)
    c = np.ones_like(G, dtype=int)
    d = np.ones_like(G, dtype=int)
    
    ######################
    # Test RestrictedShortestPath
    rsp = RestrictedShortestPath(G, c, d, 3, 0.1, 0, 3)
    print('rsp=', rsp)

    

if __name__ == "__main__":
    # TODO: Make an adjacency graph from an MDP
    Tests()
