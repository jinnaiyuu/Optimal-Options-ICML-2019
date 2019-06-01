#!/bin/python
# This code is an implementation of an approximation algorithm for solving
# Asymmetric k-center problem.
# Given a set V of n points and the distances between each pair
# the k-center problem asks to choose a subset C (subset V) of size k
# that minimizes the maximum over all points of the distance from C
# to the point.
# Asymmetric k-center is a generalization of the k-center where the distances
# are asymmetric.

import numpy as np
import math
import random
from numpy import linalg
from scipy.optimize import linprog

from ortools.constraint_solver import pywrapcp
from options.util import onehot, DeriveGraph, GetRadius

def Gamma(Gr, P, R, isPlus=True):
    """
    if isPlus=True,  Return a set of nodes reachable FROM a node in P within R steps.
    if isPlus=False, Return a set of nodes reachable TO a node in P within R steps.
    P: Set of nodes.
    """
    # print('P=', P)
    M = Gr.shape[0]
    GrI = Gr + np.identity(M)
    GrR = np.linalg.matrix_power(GrI, R) # Slow.

    if isPlus:
        nPaths = np.matmul(P, GrR)
    else:
        nPaths = np.matmul(GrR, P.transpose())
    reachable = nPaths >= 1
    ret = reachable.astype(int)

    # print('M=',M)
    # print('GrI=',GrI)
    # print('GrR=',GrR)
    # print('nPaths=',nPaths)
    # print('P   =', P)
    # print('ret =', ret)

    assert(all(P <= ret))
    return ret.flatten()

def CCV(Gr, A):
    """
    Return a center capturning vertex (CCV) in A.
    A node v is CCV iff GammaMinus(v) is a subset of GammaPlus(v).
    Output: v is represented as a onehot vector.
    """
    for i, val in enumerate(A):
        if val == 1:
            v = onehot(len(A), i)
            GammaPlus = Gamma(Gr, v, 1, True)
            GammaMinus = Gamma(Gr, v, 1, False)
            if np.all(GammaPlus >= GammaMinus):
                # v is a CCV node!
                return v

    # If no CCV node is found, return None.
    return None

def Reduce(Gr):
    M = Gr.shape[0]
    A = np.ones((M,), dtype=int)
    C = np.zeros((M,), dtype=int)

    v = CCV(Gr, A)
    # print('v=', v)
    while v is not None:
        # print("Reduce: A=", A)
        # print("Reduce: C=", C)
        # print('found CCV, v =', v)
        # TODO: v is already in the centers. How can this happen?
        C = C + v
        gammaV = Gamma(Gr, v, 2)
        # print('Gamma(v,2) =', gammaV)
        for i, val in enumerate(gammaV):
            if val == 1:
                A[i] = 0
        v = CCV(Gr, A) ########################## STUCK AT HERE
    # print("Reduce: A=", A)
    # print("Reduce: C=", C)
    gammaC = Gamma(Gr, C, 4)
    for i in range(len(A)):
        if A[i] == 1 and gammaC[i] == 1:
            A[i] = 0
            
    # print("Reduce: A=", A)
    # print("Reduce: C=", C)
    return C, A

def GetGhat(G3r, Gr, C):
    """
    Add G3r an edges of {(u, v): u in C, v in GammaPlus(Gr, C, 4)}
    """
    vs = Gamma(Gr, C, 4, True)
    ghat = np.copy(G3r)
    for i in range(len(C)):
        for j in range(len(vs)):
            if (C[i] == 1) and (vs[j] == 1) and (i is not j):
                ghat[i][j] = 1
    return ghat

def LP(Ghat, A):
    """
    Linear programming to solve Fractional set cover problem:
    min  y(V)
    s.t. y(GammaMinus(v)) >= 1 for all v in A
         y >= 0

    min  sum_{v in V} (y[v])
    s.t. for all v in A: sum_{v' in G-[v]} >= 1
         for all v in V: y[v] >= 0
    Return the assignment of y for the nodes in V. 
    """
    nV = Ghat.shape[0]

    # min y(V)
    # c: Cost function (array of ones)
    c = np.ones_like(A)
    
    # Constraint 1. 
    # y(GammaMinus(v)) >= 1 for all v in A
    constA = A.copy()
    varA = Ghat.transpose()

    # Constraint 2.
    # y >= 0
    constB = np.zeros(nV, dtype=int)
    varB = np.identity(nV)

    
    consts = np.concatenate([-constA, -constB])
    var = np.concatenate([-varA, -varB])

    res = linprog(c, A_ub=var, b_ub=consts)

    # print("Solution of LP = ", res.x)
    # print("fun = ", res.fun)
    
    return res.x, res.fun

def Vgeqi(G, C, i):
    ret = np.ones_like(C)
    gammaplus = Gamma(G, C, i-1, True)
    ret = ret - gammaplus
    return ret

def Augment(Ghat, A, C, y, p):
    """
    Augment greedly finds a set of centers to add to the already found centers C,
    using the LP-relaxed solution y.
    The algorithm here is based on ExpandingFront (Fig. 4)
    """
    i = 0
    if np.dot(y, A) < 1:
        return C
    while True:
        jmax = np.int(math.ceil(3.0 / 4.0 * p / math.pow(2, i)))
        # print('jmax=', jmax)
        for j in range(1, jmax):
            if np.dot(y, A) < 1:
                return C
            else:
                Vip1 = Vgeqi(Ghat, C, i+1)
                maxy = 0
                maxv = -1
                for i, v in enumerate(Vip1):
                    if v == 1:
                        vonehot = onehot(len(A), i)
                        gammav = Gamma(Ghat, vonehot, 1, True)
                        gammavANDA = np.minimum(gammav, A)
                        y = np.dot(y, gammavANDA)
                        if y > maxy:
                            maxy = y
                            maxv = i
                v = maxv
                C[v] = 1 # C <- C + v
                A = Vgeqi(Ghat, C, i+2)
        A = Vgeqi(Ghat, C, i+3)
        i += 1
    return C

def GreedyExpansion(D, C, k):
    """
    After guaranteeing the O(log*n) bound, add centers greedily until the number of centers reaches k.
    """
    while np.count_nonzero(C == 1) < k:
        distance = GetRadius(D.transpose(), C)
        # print('distance =', distance)
        # print('C =', C)
        minD = distance
        mins = -1
        rand = []
        for s in range(len(C)):
            if C[s] == 1:
                continue
            rand.append(s)
            Cands = np.copy(C)
            Cands[s] = 1
            newD = GetRadius(D.transpose(), Cands)
            # print('Cands =', Cands)
            # print('newD =', newD)
            if newD < minD:
                minD = newD
                mins = s
        if mins == -1:
            mins = random.choice(rand)
            
        # print(mins)
        C[mins] = 1
    # print(C)
    return C

def AKC(D, k, R):
    """
    Decision problem of asymmetric k-center.
    V: Nodes (V is implicitly represented in D.)
    D: matrix specifying a distance function V x V -> R.
    k: number of centers.
    R: maximum radius.
    """
    # print("################")
    # print("AKC: DeriveGraph(D, R)")
    Gr = DeriveGraph(D, R)
    # print("Gr = ", Gr)
    
    # print("################")
    # print("AKC: Reduce(Gr)")
    C, A = Reduce(Gr) ####################### STUCK AT HERE 
    p = 2.0 / 3.0 * (k - np.sum(C))
    # print("C = ", C)
    # print("A = ", A)
    # print("p = ", p)
    # print("################")
    # print("AKC: DeriveGraph(D, 3 * R)")
    G3r = DeriveGraph(D, 3 * R)
    # print("G3r = ", G3r)
    # print("################")
    # print("AKC: GetGhat")
    Ghat = GetGhat(G3r, Gr, C)
    # print("Ghat = ", Ghat)
    
    # print("################")
    # print("AKC: LP(Ghat, A)")
    y, yV = LP(Ghat, A)
    # print("y = ", y)
    # print("y(V) = ", yV)
    if yV > p:
        # R < R*
        # print(R, " < R*")
        return None
    else:
        p = yV
        # print("################")
        # print("AKC: Augment(Ghat, A, C, y, p)")
        C = Augment(Ghat, A, C, y, p)
        return C

def AKC_APPROX(D, k):
    R = 0
    centers = None
    while centers is None:
        R += 1
        # print("#################################################")
        # print("#################################################")
        # print("AKC: R = ", R)
        centers = AKC(D, k, R)
        # print('centers = ', centers)
    centers = GreedyExpansion(D, centers, k)
    return centers, R

def AKC_OPT(D, k):
    """
    Solve the Asymmetric k-center problem exactly using Google's OR tools.
    """
    solver = pywrapcp.Solver("Asymk")
    nNodes = D.shape[0]
    maxDistance = int(np.amax(D))
    # print("maxDistance=", maxDistance)
    centers = [solver.IntVar(0, 1) for i in range(nNodes)]
    # centers = [solver.IntVar(0, 1, "iscenter%i" % i) for i in range(nNodes)]
    minDistances = [solver.IntVar(0, maxDistance) for i in range(nNodes)]
    maxminDistance = solver.IntVar(0, maxDistance, "max min d")

    cDist = [solver.IntVar(0, maxDistance, "d(i,j)") for i in range(nNodes*nNodes)]
    for i in range(nNodes):
        for j in range(nNodes):
            # if centers[j] == 1:
            #     cDist[i, j] = D[i][j]
            # else:
            #     cDist[i, j] = maxDistance

            # (center[j] == 1 AND cDist[i,j] == D[i][j]) OR (cDist[i,j] == maxDistance)
            c = cDist[i * nNodes + j]
            d = int(D[i][j])
            center = centers[j]
            
            # solver.Add(solver.Max(solver.Min(c == d, center == 1), (c == maxDistance)) == 1)
            solver.Add(solver.Max(solver.Min(c == d, center == 1), c == maxDistance) == 1)

    for i in range(nNodes):
        solver.Add(minDistances[i] == solver.Min(cDist[i * nNodes:(i+1) * nNodes]))

    
    solver.Add(solver.Max(minDistances) == maxminDistance)

    solver.Add(solver.Sum(centers) == k)
    
    objective = solver.Minimize(maxminDistance, 1)
    variables = centers + minDistances + [maxminDistance] + cDist
    decisionBuilder = solver.Phase(variables,
                                   solver.CHOOSE_FIRST_UNBOUND,
                                   solver.ASSIGN_MIN_VALUE)
    collector = solver.LastSolutionCollector()
    collector.Add(variables)
    # collector.Add(minDistances)
    collector.AddObjective(maxminDistance)
    solver.Solve(decisionBuilder, [objective, collector])

    C = np.zeros(nNodes, dtype=int)
    # print "collector =", collector
    if collector.SolutionCount() > 0:
        bestSolution = collector.SolutionCount() - 1
        # print("maxmin d =", collector.ObjectiveValue(bestSolution))
        for i in range(nNodes):
            # print("iscenter[", i, "] =", collector.Value(bestSolution, variables[i]))
            C[i] = collector.Value(bestSolution, variables[i])
        # for i in range(nNodes):
        #     print("minDistances[", i, "] =", collector.Value(bestSolution, variables[i + nNodes]))
        R = collector.Value(bestSolution, variables[2 * nNodes])

        # for i in range(nNodes):
        #     for j in range(nNodes):
        #         print("cDist[", i, "][", j, "] =", collector.Value(bestSolution, variables[2 * nNodes + 1 + i * nNodes + j]))
                

        return C, R
    else:
        print("NO SOLUTION FOUND!")    

if __name__ == "__main__":    
    D = np.array([[0, 2, 2, 2, 2],
                  [1, 0, 1, 1, 1],
                  [2, 2, 0, 2, 2],
                  [1, 1, 1, 0, 1],
                  [0, 0, 0, 0, 0]], dtype=int)

    # D = np.array([[1, 0], [0, 0]], dtype=int)
    # d = D.transpose()
    k = 2
    
    
    C, Rbound = AKC_APPROX(D, k)
    R = GetRadius(D.transpose(), C)

    print('D=', D)
    print("Centers=", C)
    print("Radius=", R)

    print('#########')
    print('OPT')
    C, R = AKC_OPT(D, k)
    print("Centers=", C)
    print("Radius=", R)
