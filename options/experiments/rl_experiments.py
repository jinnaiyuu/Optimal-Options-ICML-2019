#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['DejaVu Sans']
import matplotlib.pyplot as plt

# Python imports.
import sys
import os
import time
import math
from collections import OrderedDict
import copy
from datetime import datetime
import argparse

# Libraries
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# simple_rl
from simple_rl.tasks import GridWorldMDP, GymMDP, TaxiOOMDP, HanoiMDP
from simple_rl.tasks.grid_world.GridWorldMDPClass import make_grid_world_from_file
from simple_rl.tasks.race_track.RaceTrackMDPClass import make_race_track_from_file
from simple_rl.planning.ValueIterationClass import ValueIteration
from simple_rl.agents import QLearningAgent, LinearQAgent, DQNAgent, RandomAgent
from simple_rl.run_experiments import run_agents_on_mdp
from simple_rl.abstraction import AbstractionWrapper, aa_helpers, ActionAbstraction, OnlineAbstractionWrapper


# options
from options.option_generation.fiedler_options import FiedlerOptions
from options.option_generation.eigenoptions import Eigenoptions
from options.option_generation.betweenness_options import BetweennessOptions
from options.option_generation.graph_drawing_options import GraphDrawingOptions
from options.graph.cover_time import ComputeCoverTime
from options.graph.spectrum import ComputeConnectivity
from options.option_generation.util import GetAdjacencyMatrix, GetIncidenceMatrix

# Q-learning
# from options.tasks.Qlearning import QLearning, RandAgent
# from options.tasks.MatrixMDP import MatrixMDP

def dump_options(filename, C, intToS):
    with open(filename, 'w') as f:
        for s in range(C.shape[0]):
            if C[s] == 1:
                state = intToS[s]
                print(state)
                w = str(state.x) + ' ' + str(state.y) + '\n'
                f.write(w)


def GetOption(mdp, k=1, sample=False, matrix=None, intToS=None, method='eigen', option_type='subgoal'):
    if matrix is not None:
        A = matrix
    elif sample:
        A, intToS = GetIncidenceMatrix(mdp)
    else:
        A, intToS = GetAdjacencyMatrix(mdp)

    if method == 'eigen':
        B, options, vectors = Eigenoptions(A, k)
    elif method == 'fiedler':
        B, options, _, vectors = FiedlerOptions(A, k)
    elif method == 'bet':
        # TODO: B is empty.
        B, options, vectors = BetweennessOptions(A, k)

        
    if not option_type == 'subgoal':
        return B, options, intToS, vectors
    # #print('knwon region=', known_region)

    egoal_list = [[]] * (len(options) * 2) 
    for i, o in enumerate(options):
        if type(o[0]) is list:
            for ss in o[0]:
                egoal_list[i * 2].append(intToS[ss])
            for ss in o[1]:
                egoal_list[i * 2 + 1].append(intToS[ss])
        else:
            egoal_list[i * 2] = [intToS[o[0]]]
            egoal_list[i * 2 + 1] = [intToS[o[1]]]

        # print('eogallist=', egoal_list[i*2])

    # for evec in evectors:
    #     print('evec=', evec)
    #     print('type(evec)=', type(evec))
        
    evector_list = [dict()] * (len(options) * 2)
    for i, o in enumerate(options):
        for j in intToS.keys():
            # print('hash(', j, ')=', hash(intToS[j]))
            # print('s[j]=', intToS[j])
            # for i in intToS[j].data.flatten():
            #     if i > 0:
            #         print(i)

            
            evector_list[i * 2][intToS[j]] = -vectors[i][j]
            evector_list[i * 2 + 1][intToS[j]] = vectors[i][j]

            # evector_list[i * 2][hash(intToS[j])] = -vectors[i][j]
            # evector_list[i * 2 + 1][hash(intToS[j])] = vectors[i][j]
            
    
    return B, egoal_list, intToS, evector_list

def GetGraphDrawingOptions(mdp, k=1):
    A, intToS = GetAdjacencyMatrix(mdp)
    B, options = GraphDrawingOptions(A, k)
    return B, options, intToS


def RunLearning(mdp, agent):
    # n_eps = 500
    n_steps = 100000

    rewards = np.zeros(n_steps)
    
    s = mdp.initial_state()
    prev_s = s
    prev_a = 0
    discount = 1
    for st in range(n_steps):            
        actions = mdp.available_actions(s)        
        a = agent.act(s, actions)
        next_s, r = mdp.next_state(s, a)

        agent.learn(s, a, r, next_s)

        rewards[st] = r # * discount
        
        if mdp.is_goal(next_s):
            s = 0
        else:
            s = next_s

    return rewards


def build_online_subgoal_option_agent(mdp, agent=QLearningAgent, n_ops=4, freqs=100, op_n_episodes=10, op_n_steps=10, method='eigen', name='-online-op'):
    goal_based_aa = ActionAbstraction(prim_actions=mdp.get_actions(), use_prims=True)

    option_agent = OnlineAbstractionWrapper(agent, agent_params={"actions":mdp.get_actions()}, action_abstr=goal_based_aa, name_ext=name, n_ops=n_ops, freqs=freqs, op_n_episodes=op_n_episodes, op_n_steps=op_n_steps, method=method, mdp=mdp)
    
    return option_agent

def build_subgoal_option_agent(mdp, subgoals, init_region, agent=QLearningAgent, vectors=None, name='-abstr', n_trajs=50, n_steps=100, classifier='list', policy='vi'):
    # print('sbugoals=', subgoals)
    goal_based_options = aa_helpers.make_subgoal_options(mdp, subgoals, init_region, vectors=vectors, n_trajs=n_trajs, n_steps=n_steps, classifier=classifier, policy=policy)
    goal_based_aa = ActionAbstraction(prim_actions=mdp.get_actions(), options=goal_based_options, use_prims=True)
    
    # num_feats = mdp.get_num_state_feats()
    option_agent = AbstractionWrapper(agent, agent_params={"actions":mdp.get_actions()}, action_abstr=goal_based_aa, name_ext=name)
        
    return option_agent


def build_point_option_agent(mdp, pairs, agent=QLearningAgent, policy='vi', name='-abstr'):
    # pairs should be a List of pair.
    # Pair is conposed of two lists.
    # Each list has init/term states.
    goal_based_options = aa_helpers.make_point_options(mdp, pairs, policy=policy)
    goal_based_aa = ActionAbstraction(prim_actions=mdp.get_actions(), options=goal_based_options, use_prims=True)

    # num_feats = mdp.get_num_state_feats()            
    option_agent = AbstractionWrapper(agent, agent_params={"actions":mdp.get_actions()}, action_abstr=goal_based_aa, name_ext=name)
    # option_agent = AbstractionWrapper(LinearQAgent, agent_params={"actions":mdp.get_actions(), "num_features":num_feats}, action_abstr=goal_based_aa, name_ext=name)
    # option_agent = AbstractionWrapper(QLearningAgent, agent_params={"actions":mdp.get_actions()}, action_abstr=goal_based_aa, name_ext=name)
    
    return option_agent


def test_utility(args, mdp):
    # The number of options to the performance
    # TODO: Compare the utility of point options vs. subgoal options?
    now_ts = str(datetime.now().timestamp())
    origMatrix, intToS = GetAdjacencyMatrix(mdp)
    known_region = list(intToS.values()) # Known region is a set of MDPStates.

    n_ops_list = [2, 4, 8, 16, 32]

    agents = []
    ql_agent = QLearningAgent(actions=mdp.get_actions())
    agents.append(ql_agent)

    method = 'fiedler'
    
    for n_ops in n_ops_list:
        _, foptions, _, fvectors = GetOption(mdp, n_ops, matrix=origMatrix, intToS=intToS, option_type=args.optiontype, method=method)
        print('#options=', n_ops)
        print(foptions)

        if args.optiontype == 'subgoal':
            known_region = list(intToS.values()) # Known region is a set of MDPStates.
            eigenoption_agent = build_subgoal_option_agent(mdp, foptions, known_region, vectors=fvectors, name='-' + method + '-' + args.optiontype + '-' + str(n_ops))
        else:
            eigenoption_agent = build_point_option_agent(mdp, foptions, agent=QLearningAgent, policy='vi', name='-' + method + '-' + args.optiontype + '-' + str(n_ops))

        
        agents.append(eigenoption_agent)

    run_agents_on_mdp(agents, mdp, instances=args.ninstances, episodes=args.nepisodes, steps=args.nsteps, open_plot=True, track_disc_reward=True, cumulative_plot=True, dir_for_plot="results/")

        
def test_offline_agent(args, mdp):
    '''
    '''
    #########################
    # Parameters for the Offline option generations
    # Incidence matrix sampling
    smp_n_traj = args.nsepisodes
    smp_steps = args.nssteps

    # Option policy learning
    op_n_episodes = args.noepisodes
    op_n_steps = args.nosteps
    
    # Final Evaluation step
    n_episodes = args.nepisodes
    n_steps = args.nsteps
    n_instances = args.ninstances
    
    n_options = args.noptions

    option_type = args.optiontype

    now = datetime.now()
    now_ts = str(now.timestamp())


    
    if args.incidence:
        origMatrix, intToS = GetIncidenceMatrix(mdp, n_traj=smp_n_traj, eps_len=smp_steps)
    else:
        origMatrix, intToS = GetAdjacencyMatrix(mdp)
    fiedlerMatrix, foptions, _, fvectors = GetOption(mdp, n_options, matrix=origMatrix, intToS=intToS, option_type=option_type, method='fiedler')
    eigenMatrix, eoptions, _, evectors = GetOption(mdp, n_options, matrix=origMatrix, intToS=intToS, option_type=option_type, method='eigen')
    _, boptions, _, bvectors = GetOption(mdp, n_options, matrix=origMatrix, intToS=intToS, option_type=option_type, method='bet')


    ######################################
    # Use the options for the learning

    vec = fvectors[0]
    # for key, items in vec.items():
    #     print('x,y=', key.x, '-', key.y)
    #     print('fval=', items)


    def ffunc(x, y):
        for key, item in vec.items():
            if key.x == x and key.y == y:
                return item
        return 0.0

    
    #################################
    # Point options
    #################################
    # eigenoption_agent = build_point_option_agent(mdp, eoptions, name='-eigen-point')
    # fiedleroption_agent = build_point_option_agent(mdp, foptions, name='-fiedler-point')
    
    
    #################################
    # Subgoal options
    #################################
    if option_type == 'subgoal':
        known_region = list(intToS.values()) # Known region is a set of MDPStates.
        # TODO: how is the state represented here in intToS?
        eigenoption_agent = build_subgoal_option_agent(mdp, eoptions, known_region, vectors=evectors, name='-eigen', n_trajs=op_n_episodes, n_steps=op_n_steps)
        fiedleroption_agent = build_subgoal_option_agent(mdp, foptions, known_region, vectors=fvectors, name='-fiedler', n_trajs=op_n_episodes, n_steps=op_n_steps)
        betoption_agent = build_subgoal_option_agent(mdp, boptions, known_region, vectors=fvectors, name='-bet', n_trajs=op_n_episodes, n_steps=op_n_steps)
    else:
        eigenoption_agent = build_point_option_agent(mdp, eoptions, agent=QLearningAgent, policy='vi', name='-eigen')
        fiedleroption_agent = build_point_option_agent(mdp, foptions, agent=QLearningAgent, policy='vi', name='-fiedler')
        betoption_agent = build_point_option_agent(mdp, boptions, agent=QLearningAgent, policy='vi', name='-bet')
    
    ql_agent = QLearningAgent(actions=mdp.get_actions(), default_q=1.0)
    rand_agent = RandomAgent(mdp.get_actions())

    # run_agents_on_mdp([ql_agent, rand_agent], mdp, instances=n_instances, episodes=n_episodes, steps=n_steps, open_plot=True, cumulative_plot=True, track_disc_reward=True, dir_for_plot="results/" + now_ts)
    run_agents_on_mdp([fiedleroption_agent, eigenoption_agent, betoption_agent, ql_agent, rand_agent], mdp, instances=n_instances, episodes=n_episodes, steps=n_steps, open_plot=True, cumulative_plot=True, track_disc_reward=True, dir_for_plot="results/", reset_at_terminal=False)
    
def test_online_agent(args, mdp):
    '''
    ''' 

   
    #########################
    # Parameters for the Offline option generations

    # Option policy learning
    op_n_episodes = args.noepisodes
    op_n_steps = args.nosteps
    n_instances = args.ninstances
    
    # Final Evaluation step
    n_episodes = args.nepisodes
    n_steps = args.nsteps

    n_options = args.noptions

    freqs = args.freqs

    # print('n_episodes=', n_episodes)
    # exit(0)
    
    now = datetime.now()
    now_ts = str(now.timestamp())
    
    ql_agent = QLearningAgent(actions=mdp.get_actions())

    rand_agent = RandomAgent(mdp.get_actions())

    # TODO: Add an arugment for selecting option generation method.
    fiedler_agent = build_online_subgoal_option_agent(mdp, agent=QLearningAgent, n_ops=n_options, freqs=freqs, op_n_episodes=op_n_episodes, op_n_steps=op_n_steps, method='fiedler', name='-fiedler')
    eigen_agent = build_online_subgoal_option_agent(mdp, agent=QLearningAgent, n_ops=n_options, freqs=freqs, op_n_episodes=op_n_episodes, op_n_steps=op_n_steps, method='eigen', name='-eigen')
    bet_agent = build_online_subgoal_option_agent(mdp, agent=QLearningAgent, n_ops=n_options, freqs=freqs, op_n_episodes=op_n_episodes, op_n_steps=op_n_steps, method='bet', name='-bet')
    
    run_agents_on_mdp([fiedler_agent, eigen_agent, ql_agent, rand_agent], mdp, instances=n_instances, episodes=n_episodes, steps=n_steps, open_plot=True, track_disc_reward=True, cumulative_plot=False, dir_for_plot="results/")

    
if __name__ == "__main__":
    
    ############################
    # Profiling: GymStateClass __eq__ is taking most of the time.
    #            Array equal is taking most of the time.
    #            How can we reduce it?
    ############################

    

    # print('neps=', args.nepisodes)
    # print('nsteps=', args.nsteps)    
    # exit()

    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment', type=str, default='online', help='online: , offline: ')
    parser.add_argument('--incidence', type=bool, default=False)
    parser.add_argument('--optiontype', type=str, default='subgoal')

    # Parameters for the task
    parser.add_argument('--task', type=str, default='grid_fourroom')
    # parser.add_argument('--rom', type=str, default='Breakout-v0', help='game to play')
    
    parser.add_argument('--nepisodes', type=int, default=10)
    parser.add_argument('--nsteps', type=int, default=10)
    parser.add_argument('--ninstances', type=int, default=1)

    # Parameters for the algorithm
    parser.add_argument('--agent', type=str, default='Q') # Q, DQN

    # Parameters for OptionsDQN
    parser.add_argument('--noepisodes', type=int, default=10)
    parser.add_argument('--nosteps', type=int, default=10)
    
    # Parameters for *online* ODQN
    parser.add_argument('--freqs', type=int, default=100)
    parser.add_argument('--noptions', type=int, default=4)
    
    # Parameters for *offline* ODQN
    parser.add_argument('--nsepisodes', type=int, default=10, help='number of episodes for incidence matrix sampling')
    parser.add_argument('--nssteps', type=int, default=10, help='number of steps for incidence matrix sampling')


    
    args = parser.parse_args()

    dom, task = args.task.split('_')
    
    if dom == 'grid':
        mdp = make_grid_world_from_file(os.path.dirname(os.path.realpath(__file__)) + '/../tasks/' + task + '.txt')
    elif dom == 'taxi':
        width = 4
        height = 4
        agent = {"x": 1, "y":1, "has_passenger":0}
        passengers = [{"x":3, "y":2, "dest_x":2, "dest_y": 3, "in_taxi":0}]
        mdp = TaxiOOMDP(width, height, agent, walls=[], passengers=passengers)
    elif dom == 'gym':
        mdp = GymMDP(env_name=task, render=False)
    elif dom == 'hanoi':
        mdp = HanoiMDP(num_pegs=3, num_discs=4)
    elif dom == 'track':
        mdp = make_race_track_from_file(os.path.dirname(os.path.realpath(__file__)) + '/../tasks/' + task + '.txt')
    else:
        print('Unknown task name: ', task)
        assert(False)

    mdp.set_gamma(0.99)

    if args.experiment == 'online':
        print('test_online_agent')
        test_online_agent(args, mdp)
    elif args.experiment == 'offline':
        print('test_offline_agent')
        test_offline_agent(args, mdp)
    elif args.experiment == 'utility':
        test_utility(args, mdp)
    else:
        print('Unregisterd experiment:', args.experiment)
        assert(False)



# def test_cover_time():
    
    # method = 'Eigen'
    # method = 'Fiedler'
    # method = 'Drawing'

    # domain = '4x4grid'
    # domain = '9x9grid'
    # domain = 'fourroom_2'    
    # fname = '../tasks/' + domain + '.txt'    
    # mdp = make_grid_world_from_file(fname)




    #######################
    # Print the cover time and the connectivity
    # origCoverTime = ComputeCoverTime(origMatrix)
    # origConnectivity = ComputeConnectivity(origMatrix)
    # print('Cover Time, Connectivity (no   op)', origCoverTime, origConnectivity)
    # opCoverTime = ComputeCoverTime(Matrix)
    # opConnectivity = ComputeConnectivity(Matrix)
    # print('Connectivity (with op)', opConnectivity)
    # print('Cover Time, Connectivity (with op)', opCoverTime, opConnectivity)
    # exit(0)
    
    #######################
    # Visualize the spectral drawing
    # Mnx = nx.to_networkx_graph(origMatrix)
    # nx.draw_spectral(Mnx)
    # plt.savefig('drawing.pdf')
    
    
    #######################
    # Visualize generated options
    # xys = []
    # # TODO: Convert the state id into
    # for o in options:
    #     init = o[0]
    #     term = o[1]
    #     print('o = ', intToS[init], intToS[term])
    #     x = intToS[init].x
    #     y = intToS[init].y
    #     xys.append((y, x))
    #     x = intToS[term].x
    #     y = intToS[term].y
    #     xys.append((y, x))
    
    # TODO: Think how to refactor the visualization part.
    # TODO: We can use the NetworkX graphics instead. 
    # mdp.visualize(xys, domain + '-' + method)



    
    #######################
    # TODO: Run the Q-learning for multiple iterations and get the average?
    # 
    #######################
    # # Q-learning on matrix
    # print('Evaluating on Q-learning.')
    # # print('matrix=', Matrix)
    # # print('origmatrix=', origMatrix)
    # 
    # # origAgent = RandAgent()
    # # opAgent = RandAgent()
    # origAgent = QLearning()
    # fiedlerAgent = QLearning()
    # eigenAgent = QLearning()
    # # mdp = MatrixMDP(origMatrix)
    # origMDP = MatrixMDP(origMatrix)
    # fiedlerMDP = MatrixMDP(fiedlerMatrix)
    # eigenMDP = MatrixMDP(eigenMatrix)
    # 
    # origRewards = RunLearning(origMDP, origAgent)
    # fiedlerRewards = RunLearning(fiedlerMDP, fiedlerAgent)
    # eigenRewards = RunLearning(eigenMDP, eigenAgent)
    # 
    # # print('orig rewards=', origRewards)
    # # print('op rewards=', opRewards)
    # 
    # #########################
    # # Take the moving average with np.convolve
    # MV_N = 5000
    # mv_origRewards = np.convolve(origRewards, np.ones((MV_N,))/MV_N, mode='valid') 
    # mv_fiedlerRewards = np.convolve(fiedlerRewards, np.ones((MV_N,))/MV_N, mode='valid')
    # mv_eigenRewards = np.convolve(eigenRewards, np.ones((MV_N,))/MV_N, mode='valid')
    # 
    # # plt.plot(origRewards, color='blue', label='noop')
    # # plt.plot(opRewards, color='red', label='fiedler')
    # plt.plot(mv_origRewards, color='blue', label='noop')
    # plt.plot(mv_fiedlerRewards, color='red', label='fiedler')
    # plt.plot(mv_eigenRewards, color='orange', label='eigen')
    # plt.legend()
    # plt.xlabel('#steps')
    # plt.ylabel('rewards')
    # 
    # # TODO: plot moving average instead
    # plt.savefig('rewards.pdf')

    # pass
