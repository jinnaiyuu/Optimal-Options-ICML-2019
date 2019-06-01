#!/bin/python

import numpy as np

#############################
# This util file should be independent to MDP module.
#############################

def onehot(length, i):
    assert(type(i) is int)
    ret = np.zeros(length, dtype=int)
    ret[i] = 1
    return ret


def neighbor(graph, n):
    '''
    Args:
        G (numpy 2d array): Adjacency matrix
        n (integer): index of the node
    Returns:
        (list of integers): neighbor nodes
    Summary:
        Given a graph adjacency matrix and a node, return a list of its neighbor nodes.
    '''
    assert(graph.ndim == 2)
    
    array = graph[n]
    # print('array=', array)
    l = []
    for i in range(len(array)):
        
        if array[i] == 1:
            l.append(i)

    # for k in range(graph.shape[0]):
    #     print('diag=', graph[k][k])
    # if i is n:
    #     print('i, n =', i, n)
    return l
    
def GetRandomWalk(G):
    """
    Given an adjacency matrix, return a random walk matrix where the sum of each row is normalized to 1
    """
    P = (G.T / G.sum(axis=1)).T
    return P

def GetRadius(D, C):
    nV = D.shape[0]

    maxd = -1
    for i in range(nV):
        mind = 10000
        for c in range(nV):
            if C[c] == 1:
                dic = D[c][i]
                if dic < mind:
                    mind = dic
        if mind > maxd:
            maxd = mind

    return maxd

def DeriveGraph(D, R):
    """
    Return Gr = (V, Er) where Er = {(u, r) : d(u, v) <= R}
    """
    Gbool = D <= R
    G = Gbool.astype(int)
    G = G - np.identity(D.shape[0])
    # print("G = ", G)
    return G

def GetCost(G):
    # TODO: Implement GetCost
    # Given an adjacency matrix, return all-pair shortest path distance
    D = np.full_like(G, -1, dtype=int)
    N = int(G.shape[0])
    
    mt = G
    distance = 1
    while distance < N:
        for x in range(N):
            for y in range(N):
                if D[x][y] == -1 and mt[x][y]:
                    D[x][y] = distance
        mt = np.matmul(mt, G)
        distance += 1

    for x in range(N):
        D[x][x] = 0
    return D

def AddEdge(G, vi, vj):
    augGraph = G.copy()
    # print('augGraph', augGraph)
    augGraph[vi, vj] = 1
    augGraph[vj, vi] = 1
    return augGraph

