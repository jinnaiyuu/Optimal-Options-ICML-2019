
# TODO: Use networkX or numpy?
import numpy as np
import networkx as nx
import scipy
from numpy import linalg
from scipy.sparse.linalg import eigsh

from options.graph.cover_time import AddEdge, ComputeCoverTime
from options.graph.spectrum import ComputeFiedlerVector, ComputeConnectivity


def Eigenoptions(G, k):
    # Generate options for smallest k eigenvectors.
    Gnx = nx.to_networkx_graph(G)
    Lscipy = nx.linalg.laplacian_matrix(Gnx).astype(float)
    # print('Laplacian', Lscipy)

    # L = Lscipy.todense().astype(float)
    # SciPy sparse matrix to Numpy matrix
    # TODO: Is this procedure taking too much time?
    evalues, evectors = eigsh(Lscipy, int(k / 2) + 1, which='SA') # 95 seconds: why is it taking more time than eig? Probably for larger matrix it is more faster? # Took 60 seconds with sparse matrix. Seems like that was the bottleneck.
    # evalues, evectors = linalg.eig(L) # 68 seconds


    # print('G=', G)
    # print('evalues', evalues)
    # print('evectors', evectors)

    options = []
    A = G.copy()
    vectors = []

    smallest_ind = np.argsort(evalues)
    
    for n in range(int(k / 2)):
        v = evectors[:, smallest_ind[n+1]]
        # print('max=', np.amax(v), ', arg=', np.argmax(v))
        # print('min=', np.amin(v), ', arg=', np.argmin(v))
        option = (np.argmax(v), np.argmin(v))
        options.append(option)
        B = AddEdge(A, option[0], option[1])
        A = B

        v = np.ravel(v).real
        # print('eigenoption: v = ', v)
        # print('real-val=', np.ravel(v).real)
        # print('type(v)=', type(v))
        vectors.append(v)
        
    return A, options, vectors

if __name__ == "__main__":
    Gnx = nx.path_graph(10)
    
    graph = nx.to_numpy_matrix(Gnx)
    
    eigenGraph, eigenOptions = Eigenoptions(graph, 8)
    print('eigenGraph', eigenGraph)
    print('eigenoptinos', eigenOptions)
    
