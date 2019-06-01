######################################################
# Suboptiaml solver for the set/vertex cover problem (CITATION)
# O(n^3) time with O(log n) suboptimality.
# We solve a linear 
# Problem:
#    minimize c^T x
#    subject to Ax >= e
# A is a set of sets. Each column represent a set of elements.
# e = (1, 1, ..., 1)
# x is a 0-1 vector with size of n
# Dual of the problem
#    maximize e^T y
#    subject to y^T A <= c 
# Js = {j | y^T Aj = cj}
# Solution <- x(Js)
# Then we greedily remove a redundant element from the solution.
#######################################################
import numpy as np
from scipy.optimize import linprog
from ortools.constraint_solver import pywrapcp
  
def find_cover(A, c):
    # TODO: Requires that m = n
    assert(A.shape[0] == A.shape[1])
    # A: m x n
    # c: n
    
    # First we solve a dual of the linear relaxation.
    dual_c = -1 * np.ones(A.shape[0]) # m
    dual_A = A.transpose()            # n x m
    dual_b = c                        # n
    print(dual_A)
    ret = linprog(c=dual_c, A_ub=dual_A, b_ub=dual_b) # linprog terminates at 1000 iterations.
    y = ret.x                         # m
    costs = np.matmul(y.transpose(), A)
    print("y=", y)
    print("costs=", costs)
    print("c=", c)
    # x(Js) will be a feasible solution with a bound of O(log n).
    Js = []
    x = np.zeros(A.shape[0], dtype=int)
    for j in range(A.shape[0]):
        if costs[j] == c[j]: 
            Js.append(j)
            x[j] = 1
    return x

def remove_redundant(A, cover):
    # Greedily remove a set from the cover.
    # b = Ax - e (b is a vector of overly covered elements)
    # Find a set in x which is a subset of b.
    # Remove it greedily.
    # Repeat this process until no set can be removed.

    c = cover
    # prev_b = None
    while True:
        b = np.matmul(A, c) - np.ones(A.shape[0])
        # print("b =", b)
        # if all(b == prev_b):
        #     assert(False)
        # prev_b = b
        removed = False
        for j in range(A.shape[1]):
            # if (A[:, j] <= b).all() and (c[j] == 1):
            if all(b - A[:, j] > 0) and (c[j] == 1):
                c[j] = 0
                removed = True
                break
        if not removed:
            # No more set can be removed
            print("No more set can be removed.")
            break
                
    return c

def SC_APPROX2(X):
    # V. Chvatal 1979
    m = X.shape[0]
    n = X.shape[1]
    J = np.zeros(m, dtype=int) # number of subsets
    P = np.zeros(n, dtype=int) # number of elements
    while any(P == 0):
        # print("P =", P)
        # print("J =", J)
        maxCover = -1
        cover = -1
        for i in range(m):
            c = 0
            for j in range(n):
                if P[j] == 0 and X[i][j] == 1:
                    c += 1
            if c > maxCover:
                maxCover = c
                cover = i
        for j in range(n):
            P[j] = max(P[j], X[cover][j])
        J[cover] = 1
    return J

def SC_APPROX(X):
    # Dorit S. Hochbaum 1982
    c = np.ones(X.shape[1], dtype=int)
    cover = find_cover(X, c)
    print('cover (redn) =', cover)
    C = remove_redundant(X, cover)
    print('cover        =', C)
    return C

def SC_OPT(X):
    print("X=", X)
    solver = pywrapcp.Solver("SetCover")
    nSubsets = int(X.shape[0])
    nElements = int(X.shape[1])
    cover = [solver.IntVar(0, 1) for i in range(nSubsets)]
    coverSize = solver.Sum(cover[i] for i in range(nSubsets))

    doesCover = [[solver.IntVar(0, 1) for j in range(nElements)] for i in range(nSubsets)]
    isCovered = [solver.IntVar(0, nElements) for i in range(nSubsets)]
    
    for i in range(nElements):
        for j in range(nSubsets):
            v = int(X[j][i])
            solver.Add(doesCover[j][i] == solver.Min(cover[j], v))
            
        solver.Add(isCovered[i] == solver.Sum(doesCover[j][i] for j in range(nSubsets)))
        solver.Add(isCovered[i] >= 1)
        
    objective = solver.Minimize(coverSize, 1)
    variables = cover + [n for sublist in doesCover for n in sublist] + isCovered + [coverSize]
    decisionBuilder = solver.Phase(variables,
                                   solver.CHOOSE_FIRST_UNBOUND,
                                   solver.ASSIGN_MIN_VALUE)
    collector = solver.LastSolutionCollector()
    collector.Add(variables)
    collector.AddObjective(coverSize)
    solver.Solve(decisionBuilder, [objective, collector])

    C = np.zeros(nSubsets, dtype=int)
    if collector.SolutionCount() > 0:
        bestSolution = collector.SolutionCount() - 1
        for i in range(nSubsets):
            C[i] = collector.Value(bestSolution, variables[i])
        return C
    else:
        print("NO SOLUTION FOUND!")    
    

if __name__ == "__main__":
    # Main function
    A = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1]])
    
    c = np.array([1, 1, 1])

    opt = SC_OPT(A)
    approx = SC_APPROX2(A)
    print("opt = ", opt)
    print("approx = ", approx)
    
