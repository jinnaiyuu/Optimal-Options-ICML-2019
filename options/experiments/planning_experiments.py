#!/usr/bin/env python

# Python imports.
import sys
import os
import time
from collections import OrderedDict
import copy

# Libraries
import numpy as np
import matplotlib.pyplot as plt

# Other imports.
from simple_rl.tasks import GridWorldMDP
from simple_rl.tasks.grid_world.GridWorldMDPClass import make_grid_world_from_file
from simple_rl.tasks import TaxiOOMDP
from simple_rl.planning.ValueIterationClass import ValueIteration

# Options
from options.option_generation.MIMO import MIMO
from options.option_generation.MOMI import MOMI

from options.option_generation.vi_distance import get_distance
    
def plot(approx, opt, domain, problem, xrange):
    yrange = range(min(min(approx),min(opt)), max(max(approx),max(opt)) + 1)
    plt.yticks(yrange)
    plt.xticks(xrange)
    plt.plot(xrange, approx, '.-')
    
    plt.plot(xrange[0:len(opt)], opt, '.-')
    # plt.legend(['APPROX'])
    plt.legend(['APPROX', 'OPT'])
    if problem == 'MIMO':
        plt.ylabel('#Iterations') # iterations vs. backups
        plt.xlabel('#Options')
    elif problem == 'MOMI':
        plt.xlim(max(xrange), min(xrange))
        plt.xlabel('#Iterations') # iterations vs. backups
        plt.ylabel('#Options')
    else:
        print('Undefined problem', problem)
        assert(False)
    plt.title(domain)
    # plt.show()
    plt.savefig(os.path.dirname(os.path.realpath(__file__)) + '/../figures/' + problem + '/' + domain + '.pdf')
    plt.close()

def dump_options(filename, C, intToS):
    with open(filename, 'w') as f:
        for s in range(C.shape[0]):
            if C[s] == 1:
                state = intToS[s]
                print(state)
                w = str(state.x) + ' ' + str(state.y) + '\n'
                f.write(w)
    

def run_MIMO(domain, mdp, num_options, opt_num_options=None):
    if opt_num_options is None:
        opt_num_options = num_options
    sIndex, intToS, distance = get_distance(mdp) # Distance function implemented as a 2d array
    
    v0 = distance.max()
    approx_v = [v0]
    # approx_b = [b0]
    opt_v = [v0]
    # opt_b = [b0]
    for op in num_options:
        C, va = MIMO(mdp, distance, k=op, solver='archer')
        dump_options('data/' + domain + '_' + str(op) + 'ops_' + 'archer', C, intToS)
        approx_v.append(va)
    for op in opt_num_options:
        # approx_b.append(ba)
        C, vo = MIMO(mdp, distance, k=op, solver='optimal')
        dump_options('data/' + domain + '_' + str(op) + 'ops_' + 'optimal', C, intToS)
        opt_v.append(vo)
        # opt_b.append(bo)

    print("approx", approx_v)
    print("opt", opt_v)

    # exit(0)

    xvals = copy.copy(num_options)
    xvals.insert(0, 0)
    
    # Plot the figure on #Iterations
    plot(approx_v, opt_v, domain, 'MIMO', xvals)

    
def run_MOMI(domain, mdp, num_iters, op_num_iters=None):
    if op_num_iters is None:
        op_num_iters = num_iters
        
    sToInt, intToS, distance = get_distance(mdp) # Distance function implemented as a 2d array

    
    approx_op = [0]
    opt_op = [0]
    for iters in num_iters:
        apC = MOMI(mdp, distance, l=iters, solver='chvatal')        
        approx_op.append(int(np.sum(apC)))
        print('APPROX options')
        dump_options(os.path.dirname(os.path.realpath(__file__)) + '/../data/' + domain + '_' + str(iters) + 'iter_' + 'chvatal', apC, intToS)
        
                
    for iters in op_num_iters:
        opC = MOMI(mdp, distance, l=iters, solver='optimal')
        opt_op.append(int(np.sum(opC)))
        print('OPT options')
        dump_options(os.path.dirname(os.path.realpath(__file__)) + '/../data/' + domain + '_' + str(iters) + 'iter_' + 'optimal', opC, intToS)

    # number of iterations without options
    v0 = distance.max()
    xvals = copy.copy(num_iters)
    xvals.insert(0, v0)
                
    print("approx", approx_op)
    print("opt", opt_op)
    print('iters', xvals)
 
    # exit(0)
    
    # Plot the figure on #Iterations
    plot(approx_op, opt_op, domain, 'MOMI', xvals)
   
if __name__ == "__main__":
    # ortools (https://developers.google.com/optimization/) is required to solve MOMI optimally.
    # scipy is required to run approximation algorithms.
    domain = '9x9grid'
    # domain = '5x5grid'
    # domain = '3x3grid'
    
    fname = os.path.dirname(os.path.realpath(__file__)) + '/../tasks/' + domain + '.txt'
    mdp = make_grid_world_from_file(fname)

    print('MDP =', mdp)


    run_MOMI(domain, mdp, [6])
