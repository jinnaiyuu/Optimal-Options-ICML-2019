
# TODO: Use networkX or numpy?
import numpy as np

def GenerateRandomConnectedGraph(size, edge_dense):
    # 1. Add a node one by one to the graph connecting to a random node.
    # 2. Add edges randomly until the density is reached.
    G = np.zeros((size, size), dtype=int)
    
    for n in range(size):
        if n == 0:
            continue
        # randomly pick a node in graph to connect to the new node.
        v = np.random.randint(0, n)
        G[n][v] = 1
        G[v][n] = 1

    m = n - 1
    while m < size * edge_dense:
        # Add an edge randomly
        s = np.random.randint(0, size)
        t = np.random.randint(0, size)
        if G[s][t] == 0:
            continue
        else:
            G[s][t] = 1
            G[t][s] = 1
            m += 1
    return G


if __name__ == "__main__":
    pass
