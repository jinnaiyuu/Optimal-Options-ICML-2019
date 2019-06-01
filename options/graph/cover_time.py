import random
import numpy as np
import networkx as nx
from options.graph.spectrum import ComputeFiedlerVector, ComputeConnectivity
from options.util import AddEdge, neighbor

# def HittingTime(G, t):
#     """
#     Given a graph adjacency matrix and a start and goal state,
#     return a hitting time.
#     """
#     A = G.copy()
#     A[t, :] = 0
#     A[t, t] = 1
# 
#     # print('A', A)
#     A = (A.T / A.sum(axis=1)).T
#     # print('A', A)
#     B = A.copy()
#     Z = []
#     for n in range(G.shape[0] * G.shape[0] * 2):
#         Z.append(B[:, t]) # TODO: We can get the whole vector B[:, t] to speedup by n times
#         B = np.dot(B, A)
# 
#     ret = np.zeros_like(Z[0])
#     for n in range(len(Z)):
#         if n == 0:
#             ret += Z[n] * (n+1)
#         else:
#             ret += (Z[n] - Z[n-1]) * (n+1)
#     if any(Z[len(Z) - 1] < 1):
#         ret += (1 - Z[len(Z)-1]) * (len(Z))
#     # print('Z', Z)
#     # print('ret', ret)
#     return ret

def ComputeCoverTimeS(G, s, sample=1000):
    ##########################
    # PLEASE WRITE A TEST CODE
    ##########################
    '''
    Args:
        G (numpy 2d array): Adjacency matrix (may be an incidence matrix).
        s (integer): index of the initial state
        sample (integer): number of trajectories to sample
    Returns:
        (float): the expected cover time from state s
    Summary:
        Given a graph adjacency matrix, return the expected cover time starting from node s. We sample a set of trajectories to get it.
    '''
    
    N = G.shape[0]

    n_steps = []
    
    for i in range(sample):
        visited = np.zeros(N, dtype=int)
        visited[s] = 1
        cur_s = s
        cur_steps = 0

        while any(visited == 0):
            s_neighbor = neighbor(G, cur_s)
            next_s = random.choice(s_neighbor)
            visited[next_s] = 1
            cur_s = next_s
            cur_steps += 1
            
        n_steps.append(cur_steps)

    # print('n_steps=', n_steps)

    avg_steps = sum(n_steps) / sample
    return avg_steps

def ComputeCoverTime(G, samples=1000):
    ##########################
    # PLEASE WRITE A TEST CODE
    ##########################
    '''
    Args:
        G (numpy 2d array): Adjacency matrix (or incidence matrix)
    Returns:
        (float): the expected cover time
    Summary:
        Given a graph adjacency matrix, return the expected cover time.
    '''
    N = G.shape[0]

    c_sum = 0

    for i in range(samples):
        init = random.randint(0, N-1)
        c_i = ComputeCoverTimeS(G, init, sample=1)
        c_sum += c_i
        
    return float(c_sum) / float(samples)

if __name__ == "__main__":

    # PlotConnectivityAndCoverTime(100)
    # exit(0)
    
    Gnx = nx.path_graph(4)
    
    graph_ = nx.to_numpy_matrix(Gnx)
    graph = np.asarray(graph_)

    v = ComputeFiedlerVector(Gnx) # numpy array of floats
    
    augGraph = AddEdge(graph, np.argmax(v), np.argmin(v))
    

    # print('Graphs')
    # print(graph)
    # print(augGraph)
    t2 = ComputeCoverTime(augGraph)
    print('CoverTime Aug1', t2)
    lb2 = nx.algebraic_connectivity(nx.to_networkx_graph(augGraph))
    print('lambda        ', lb2)

    
