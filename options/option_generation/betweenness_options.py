import networkx as nx
from collections import defaultdict
import numpy as np

def BetweennessOptions(T, k, t=0.1):
    '''
    Args:
       T: transition matrix
       k: number of options to generate
    '''
    
    # print("find betweenness options...")
    G = nx.from_numpy_matrix(T) 
    N = G.number_of_nodes()
    M = G.number_of_edges()
    # print("nodes=", N)
    # print("edges=", M)

    #########################
    ## 1. Enumerate all candidate subgoals
    #########################
    subgoal_set = []
    for s in G.nodes():
        # print("s=", s)
        csv = nx.betweenness_centrality_subset(G, sources=[s], targets=G.nodes())
        # csv = nx.betweenness_centrality(G)
        # print("csv=", csv)
        for v in csv:
            if (s is not v) and (csv[v] / (N-2) > t) and (v not in subgoal_set):
                subgoal_set.append(v)

    if len(subgoal_set) > int(k/2):
        subgoal_set = subgoal_set[0:int(k/2)]

    # for s in subgoal_set:
    #     print(s, " is subgoal")
    # n_subgoals = sum(subgoal_set)
    # print(n_subgoals, "goals in total")
    # centralities = nx.betweenness_centrality(G)
    # for n in centralities:
    #     print("centrality=", centralities[n])

    #########################
    ## 2. Generate an initiation set for each subgoal
    #########################
    initiation_sets = defaultdict(list)
    support_scores = defaultdict(float)
    
    for g in subgoal_set:
        # TODO: This computation is taking most of the time.
        csg = nx.betweenness_centrality_subset(G, sources=G.nodes(), targets=[g])
        score = 0
        for s in G.nodes():
            if csg[s] / (N-2) > t:
                initiation_sets[g].append(s)
                score += csg[s]
        support_scores[g] = score
                
    # for g in subgoal_set:
    #     print("init set for ", g, " = ", initiation_sets[g])

    #########################
    ## 3. Filter subgoals according to their supports
    #########################
    filtered_subgoals = []

    subgoal_graph = G.subgraph(subgoal_set)
    
    sccs = nx.connected_components(subgoal_graph) # TODO: connected components are used instead of SCCs
    # sccs = nx.strongly_connected_components(G)
    for scc in sccs:
        scores = []
        goals = []
        for n in scc:
            scores.append(support_scores[n])
            goals.append(n)
            # print("score of ", n, " = ", support_scores[n])
        # scores = [support_scores[x] for x in scc]
        best_score = max(scores)
        best_goal = goals[scores.index(best_score)]
        filtered_subgoals.append(best_goal)

    options = []
    for g in filtered_subgoals:
        init_set = initiation_sets[g]
        goal_set = []
        goal_set.append(g)
        options.append((init_set, goal_set))

    vectors = []
    for o in options:
        v = np.zeros(T.shape[0], dtype=float)
        for s in o[1]:
            v[s] = 1.0
        vectors.append(v)
    # print("done.")
    return T, options, vectors
