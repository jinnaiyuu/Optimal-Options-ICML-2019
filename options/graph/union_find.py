import numpy as np

def TestUnionFind():
    # Test UnionFind
    V = [0, 1, 2, 3]
    E = [[0, 1], [2, 3], [1, 2]]

    clusters = UnionFind(V, E)
    print('clusters=', clusters)
    
def UnionFind(V, E):
    # V: a list of nodes
    # E: a list of pair of nodes 
    # Return a set of connected components in the graph

    for e in E:
        assert(e[0] in V)
        assert(e[1] in V)
    
    clusters = []
    for v in V:
        clusters.append([v])

    for e in E:
        c1 = None
        c2 = None
        # print('e=', e)
        for c in clusters:
            if e[0] in c:
                c1 = c
            elif e[1] in c:
                c2 = c
        assert(c1 is not None)
        assert(c2 is not None)
        new_c = c1 + c2
        clusters.remove(c1)
        clusters.remove(c2)
        clusters.append(new_c)
    return clusters

if __name__ == '__main__':
    TestUnionFind()
