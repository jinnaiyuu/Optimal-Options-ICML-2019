import math
import numpy as np
import copy
from collections import defaultdict
from options.graph.restricted_shortest_path import RestrictedShortestPath
from options.graph.matching import ApproxMinimumWeightMatching, GreedyMinimumWeightMatching

# An approximation algorithm for Diameter-constrained Steiner tree proposed in Marathe et al. 1998.

def MinimumWeightMatching(G, edge_weight, method='GW'):
    # TODO: First implement an optimal algorithm?
    if method == 'GW':
        return ApproxMinimumWeightMatching(G, edge_weight)
    elif method == 'greedy':
        return GreedyMinimumWeightMatching(G, edge_weight)
    else:
        print('No matching algorithm', method)
        assert(False)

def DiameterConstrainedSteinerTree(G, c, d, K, D, eps):
    # G   := Numpy matrix (N x N). 1 if edge exists
    # c   := Numpy matrix (N x N). Each value represents its edge cost.
    # d   := Numpy matrix (N x N). Each value represents its edge cost.
    # K   := numpy array (N). 1 if the node is in terminals.
    # D   := A bound on the Diameter
    # eps := real value

    N = G.shape[0]
    
    # 1. Initialize the set of clusters C to contain |K| singleton sets, one for each terminal in K.
    #    For each cluster in C, define the single node in the cluster to be the center for the cluster.
    clusters = []
    for i in range(len(K)):
        if K[i] == 1:
            clusters.append([i])

    phase_count = 1

    sTree = np.full((N, N), -1, dtype=int)
    options = []

    while len(clusters) > 1:
        # 2. Repeat until there remains a single cluster in C.
        # (a) Let the set of clusters Ci at the beginning of the i-th phase.            
        # (b) Construct a complete graph G as follows: The node set Vi of Gi is {v: v is the center of a cluster in C}.
        # Let path Pxy be a (1+eps)-approximation to the minimum c-cost diameter D-bounded path between centers vx and vy in G.
        # Between every pair of ndoes vx and vy in Vi, include an edge (vx, vy) in Gi of weight equal to the c-cost of Pxy.
        print('Phase', phase_count)
        print('#clusters=', len(clusters))
        centers = []
        for cls in clusters:
            centers.append(cls[0])

        clique = np.full((len(centers), len(centers)), 1000000, dtype=int)
        # print('shape', np.ones((len(centers), len(centers)), dtype=int).shape)
        # print('shape', np.identity(len(centers), dtype=int).shape)
        clq_u = np.subtract(np.ones((len(centers), len(centers)), dtype=int), np.identity(len(centers), dtype=int))

        for x in range(len(centers)):
            for y in range(x + 1, len(centers)):
                vx = centers[x]
                vy = centers[y]
        
                cPxy = RestrictedShortestPath(G, c, d, D, eps, vx, vy)
                clique[x][y] = cPxy
                clique[y][x] = cPxy
        
        
        # (c) Find a minimum-weight matching of largest cardinality in Gi
        _, matching, _ = MinimumWeightMatching(clq_u, clique)
        
        # (d) For each edge e = (vx, vy) in the matching, merge cluster Cx and Cy.
        isIncluded = np.zeros(len(clusters), dtype=int)
        new_clusters = []
        for m in matching:
            vx = m[0]
            vy = m[1]
            n_cluster = clusters[vx] + clusters[vy]
            sTree[clusters[vx][0]][clusters[vy][0]] = clique[vx][vy]
            sTree[clusters[vy][0]][clusters[vx][0]] = clique[vx][vy]
            options.append((clusters[vx][0], clusters[vy][0]))
            new_clusters.append(n_cluster)
            isIncluded[vx] = 1
            isIncluded[vy] = 1

        if np.any(isIncluded == 0):
            lonelyCluster = np.argwhere(isIncluded == 0)[0][0]
            new_clusters.append(clusters[lonelyCluster])

        clusters = new_clusters
        phase_count += 1
    
    return sTree, options

    

if __name__ == "__main__":
    # TODO: Make an adjacency graph from an MDP
    
    
    G = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]], dtype=int)
    c = np.ones_like(G, dtype=int)
    d = np.ones_like(G, dtype=int)
    # c = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int) # c should be the cost of adding options (fill with 1)
    # d = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int) # d should be the action cost (let's fill with 1 for now)
    K = np.array([1, 0, 1, 1], dtype=int)

    ######################
    # Test DiameterConstrainedSteinerTree(G, c, d, K, D, eps)
    tree, options = DiameterConstrainedSteinerTree(G, c, d, K, 2, 0.01)
    print('Tree=', tree)
    print('Options', options)

        

# def ShortestPathTree(graph, edge_weight, root):
#     # TODO: Does triangle inequality holds true?
#     # Dijkstra algorithm
#     N = int(graph.shape[0])
# 
#     closedList = np.zeros(N, dtype=int)
#     queue = []
#     queue.append(root)
#     closedList[root] = 1
# 
#     edgeList = np.zeros_like(graph, dtype=int)
#     
#     while len(queue) > 0:
#         node = queue.pop()
#         neighbors = Neighbor(graph, node)
#         for i in neighbors:
#             if closedList[i] == 0:
#                 queue.append(i)
#                 closedList[i] = 1
#     
#     pass
