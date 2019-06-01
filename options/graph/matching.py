import numpy as np
from simple_rl.tasks import GridWorldMDP
from simple_rl.tasks.grid_world.GridWorldMDPClass import make_grid_world_from_file
# from options_experiments import GetAdjacencyMatrix, GetCost
from options.graph.union_find import UnionFind

def ForestProblem(G, c, f):
    # G: graph adjacency matrix
    # c: cost function E -> Q+
    # f: proper function, a set 
    N = G.shape[0]

    F = [] # empty set
    LB = 0.0
    clusters = [] # set of singleton sets

    for i in range(N):
        cluster = [i]
        clusters.append(cluster)

    assert(len(clusters) == N)
    
    d = np.zeros(N, dtype=int)

    num_iter = 1
    while any(f(cluster) == 1 for cluster in clusters) and len(clusters) > 1:
        print('iter', num_iter, 'clusters=', clusters)
        min_eps = 10000000000
        min_pair = None
        clst_pair = None
        for p in range(len(clusters)):
            for q in range(p + 1, len(clusters)):
                assert(p is not q)
                for i in clusters[p]:
                    for j in clusters[q]:
                        if G[i][j] == 0:
                            continue
                        eps = (c[i][j] - d[i] - d[j]) / (f(clusters[p]) + f(clusters[q]))
                        
                        if eps < min_eps:
                            min_eps = eps
                            min_pair = (i, j)
                            clst_pair = (p, q)
        eps = min_eps
        F.append(min_pair)
        for cluster in clusters:
            for v in cluster:
                d[v] += eps * f(cluster)

        for cluster in clusters:
            LB += eps * f(cluster)

        c1 = clusters[clst_pair[0]]
        c2 = clusters[clst_pair[1]]
        new_c = c1 + c2
        clusters.remove(c1)
        clusters.remove(c2)
        clusters.append(new_c)
        
        num_iter += 1
    # TODO: F <- {e in F: For some connected component N of (V, F - {e}), f(N) = 1}
    print('F=', F)
    Fdash = []
    V = range(N)
    for e in F:
        fmine = F.copy()
        fmine.remove(e)
        components = UnionFind(V, fmine)
        if any(f(comp) == 1 for comp in components):
            Fdash.append(e)
    return Fdash, LB

def ApproxMinimumWeightMatching(G, c):
    # G: graph adjacency matrix
    # c: cost function E -> Q+
    def f(S):
        if len(S) % 2 == 0:
            return 0
        else:
            return 1
        
    # TODO: Add one node
    N = G.shape[0]

    if N % 2 == 0:
        edges, LB = ForestProblem(G, c, f)
    else:
        Gexp = np.ones((N+1, N+1), dtype=int)
        cexp = np.full((N+1, N+1), 10000000, dtype=int)
        for i in range(N):
            for j in range(N):
                Gexp[i][j] = G[i][j]
                cexp[i][j] = c[i][j]
        edges, LB = ForestProblem(Gexp, cexp, f)
        for e in edges:
            if any(v == N for v in e):
                edges.remove(e)
                break
        LB = LB - 10000000
    aMatrix = np.zeros_like(G, dtype=int)
    for e in edges:
        aMatrix[e[0]][e[1]] = 1
        aMatrix[e[1]][e[0]] = 1

    return aMatrix, edges, LB

def GreedyMinimumWeightMatching(G, edge_weight):
    ############################
    # TODO: Read Goemans&Williamson'95 (approx. algorithm)
    

    ############################
    # Greedy algorithm: The procedure has a tight performance guarantee of 4/3 * n^{0.585} (Reingold&Tarjan 1981).
    # TODO: I don't see how this procedure can have a performance guarantee. One can generate an arbitrary bad weights.
    N = edge_weight.shape[0]
    nEdges = int(math.floor(N / 2))
    
    solution = np.zeros_like(edge_weight)
    edge_list = []
    connected_nodes = np.zeros(N, dtype=int)

    E = edge_weight.copy()    
    n = 0
    while n < nEdges:
        # TODO: Remove connected nodes
        minEdge = np.unravel_index(np.argmin(E), E.shape)

        # print('edge_weight', edge_weight)
        # print('minEdge=', minEdge)
        solution[minEdge[0], minEdge[1]] = 1
        solution[minEdge[1], minEdge[0]] = 1
        edge_list.append((minEdge[0], minEdge[1]))

        connected_nodes[minEdge[0]] = 1
        connected_nodes[minEdge[1]] = 1

        E[minEdge[0],:] = 100000000
        E[:,minEdge[0]] = 100000000
        E[minEdge[1],:] = 100000000
        E[:,minEdge[1]] = 100000000
        n += 1


    # Matrix and pair of edges
    return solution, edge_list, 0
    


def Tests():    
    ######################
    # Test MinimumWeightMatching
    G = np.array([[0, 1, 1, 1],
                  [1, 0, 1, 1],
                  [1, 1, 0, 1],
                  [1, 1, 1, 0]], dtype=int)
    costs = np.array([[100000000, 1, 2, 2], [1, 10000000, 2, 2], [2, 2, 10000000, 10], [2, 2, 10, 10000000]], dtype=int)
    mwm = ApproxMinimumWeightMatching(G, costs)
    print('mwm=', mwm)


if __name__ == '__main__':
    G = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]], dtype=int)
    c = np.array([[1, 2, 1, 1],
                  [2, 1, 1, 1],
                  [1, 1, 1, 2],
                  [1, 1, 2, 1]]
                 , dtype=int)
    
    F, options, LB = ApproxMinimumWeightMatching(G, c)

    print('F\'=', F)
    print('options=', options)
    print('LB=', LB)
