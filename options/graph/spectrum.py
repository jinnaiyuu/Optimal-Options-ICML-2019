import numpy as np
import networkx as nx


def ComputeFiedlerVector(G):
    """
    Given a graph adjacency matrix, return a Fielder vector.
    """
    # TODO: implement a case where it converts to a networkx graph if G is a numpy array
    if type(G) == type(np.ndarray((1, 1, 1), dtype=float)):
        G = nx.to_networkx_graph(G)
        
    v = nx.fiedler_vector(G)
    
    return v

def ComputeConnectivity(G):
    lb = nx.algebraic_connectivity(nx.to_networkx_graph(G))
    # print('lambda        ', lb)
    return lb

if __name__ == "__main__":
    pass
