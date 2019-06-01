#!/usr/bin/env python

# Python imports.
import sys
import time
from collections import OrderedDict, defaultdict
import numpy as np

# Other imports.
from simple_rl.tasks import GridWorldMDP
from simple_rl.planning.ValueIterationClass import ValueIteration
from options.option_generation.ValueIterationDist import ValueIterationDist

def get_distance(mdp, epsilon=0.05):
    
    vi = ValueIteration(mdp)
    vi.run_vi()
    vstar = vi.value_func # dictionary of state -> float

    states = vi.get_states() # list of state

    distance = defaultdict(lambda: defaultdict(float))
    
    v_df = ValueIterationDist(mdp, vstar)
    v_df.run_vi()
    d_to_s = v_df.distance
    for t in states:
        for s in states:
            distance[t][s] = max(d_to_s[t] - 1, 0)

    for s in states: # s: state
        vis = ValueIterationDist(mdp, vstar)
        vis.add_fixed_val(s, vstar[s])
        vis.run_vi()
        d_to_s = vis.distance
        for t in states:
            distance[t][s] = min(d_to_s[t], distance[t][s])


    sToInd = OrderedDict()
    indToS = OrderedDict()
    for i, s in enumerate(states):
        sToInd[s] = i
        indToS[i] = s

    d = np.zeros((len(states), len(states)), dtype=int)
    # print "type(d)=", type(d)
    # print "d.shape=", d.shape
    for s in states:
        for t in states:
            # print 's, t=', index[s], index[t]
            d[sToInd[s]][sToInd[t]] = distance[s][t]
    
    return sToInd, indToS, d


if __name__ == "__main__":
    mdp = GridWorldMDP(width=5, height=5, init_loc=(1, 1), goal_locs=[(5, 5)])
    sToInd, indToS, d = get_distance(mdp)
