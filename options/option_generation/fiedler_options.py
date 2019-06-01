import numpy as np
from numpy import linalg
import networkx as nx
import matplotlib.pyplot as plt
from options.graph.cover_time import AddEdge, ComputeCoverTime
from options.graph.spectrum import ComputeFiedlerVector, ComputeConnectivity


def FiedlerOptions(G, k, subgoal=False):
    no = 0

    X = nx.to_networkx_graph(G)
    if not nx.is_connected(X):
        cs = list(nx.connected_components(X))
        for c_ in cs:
            if len(c_) > 1:
                c = c_
                break
        Xsub = X.subgraph(c)
        A = nx.to_numpy_matrix(Xsub)
        print('connected comp =', c)
    else:
        A = G.copy()

    options = []
    
    eigenvalues = []
    eigenvectors = []
    
    while no < k:
        v = ComputeFiedlerVector(nx.to_networkx_graph(A))
        lmd = ComputeConnectivity(A)

        # maxv = np.amax(v)
        # maxs = []
        # for i, val in enumerate(v):
        #     if val > maxv - 0.02:
        #         maxs.append(i)
        #         
        # minv = np.argmin(v)
        # mins = []
        # for i, val in enumerate(v):
        #     if val < minv + 0.02:
        #         mins.append(i)
        # 
        # print('maxs=', maxs)
        maxs = [np.argmax(v)]
        mins = [np.argmin(v)]
        option = (maxs, mins)
        
        options.append(option)
        if subgoal:
            B = A.copy()
            B[:, option[1][0]] = 1
            B[option[1][0], :] = 1
        else:
            B = AddEdge(A, option[0][0], option[1][0])
        A = B
        no += 2
        eigenvalues.append(lmd)
        eigenvectors.append(v)

    # TODO: If A is a subgraph of G, convert the acquired eigenvectors to the original size.
    if not nx.is_connected(X):
        evecs = []
        for v in eigenvectors:
            newv = np.zeros(G.shape[0])
            i = 0
            j = 0
            while i < A.shape[0]:
                if j in c:
                    newv[j] = v[i]
                    i += 1
                j += 1
            evecs.append(newv)
    else:
        evecs = eigenvectors

    return A, options, eigenvalues, evecs

if __name__ == "__main__":

    Gnx = nx.path_graph(10)
    
    graph = nx.to_numpy_matrix(Gnx)

    proposedAugGraph, options = FiedlerOptions(graph, 8)

    pGnx = nx.to_networkx_graph(proposedAugGraph)
    
    nx.draw_spectral(pGnx)
    plt.savefig('drawing.pdf')

    
    t = ComputeCoverTime(graph)
    print('CoverTime     ', t)
    lb = nx.algebraic_connectivity(nx.to_networkx_graph(graph))
    print('lambda        ', lb)

    t3 = ComputeCoverTime(proposedAugGraph)
    print('CoverTime Aug ', t3)
    lb3 = nx.algebraic_connectivity(nx.to_networkx_graph(proposedAugGraph))
    print('lambda        ', lb3)

    
