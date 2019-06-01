#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Python imports.
import sys
import logging
import argparse
import numpy as np
import tensorflow as tf
import argparse
import random

from simple_rl.agents import RandomAgent
from simple_rl.tasks import GymMDP, PinballMDP
from simple_rl.tasks.gym.GymStateClass import GymState
from simple_rl.agents.func_approx.ExperienceBuffer import ExperienceBuffer

from simple_rl.tasks import GridWorldMDP
from simple_rl.tasks.grid_world.GridWorldMDPClass import make_grid_world_from_file

from options.OptionWrapper import OptionWrapper, CoveringOption
from options.OptionAgent import OptionAgent

def arguments():
    parser = argparse.ArgumentParser()

    # pinball files = pinball_box.cfg  pinball_empty.cfg  pinball_hard_single.cfg  pinball_medium.cfg  pinball_simple_single.cfg
    
    parser.add_argument('--rseed', type=int, default=1234)

    # Experiment type
    parser.add_argument('-e', '--exp', type=str, choices=["sample", "visop", "vistraj", "visterm", "visvis", "visfval", "evaloff", "evalon", "generate", "train"])
    
    # Parameters for the task
    parser.add_argument('--tasktype', type=str, default='pinball')
    parser.add_argument('--task', type=str, default='pinball_empty.cfg')
    parser.add_argument('--base', action='store_true')
    
    parser.add_argument('--nepisodes', type=int, default=100)
    parser.add_argument('--nsteps', type=int, default=200)
    parser.add_argument('--ninstances', type=int, default=1)

    # Option generation
    parser.add_argument('--snepisodes', type=int, default=100)
    parser.add_argument('--snsteps', type=int, default=100)
    parser.add_argument('--restoretraj', action='store_true')
    parser.add_argument('--trajdir', type=str, default="__default")
    parser.add_argument('--sptrainingstep', type=int, default=100)

    # Parameters for the Agent
    parser.add_argument('--noptions', type=int, default=1)

    parser.add_argument('--initall', type=bool, default=True)
    parser.add_argument('--reverse', action='store_true')

    # TODO: Buffer size may be too small. Batch size may be too large.
    parser.add_argument('--buffersize', type=int, default=512)
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--lowupdatefreq', type=int, default=32)
    
    parser.add_argument('--obuffersize', type=int, default=512)
    parser.add_argument('--obatchsize', type=int, default=128)
    parser.add_argument('--highupdatefreq', type=int, default=32)
    
    parser.add_argument('--ofreq', type=int, default=256) # Online-generation agent
    parser.add_argument('--ominsteps', type=int, default=512) # Online-generation agent

    parser.add_argument('--highmethod', type=str, default='linear')
    parser.add_argument('--lowmethod', type=str, default='linear')
    parser.add_argument('--ffunction', type=str, default='nn')

    # Parameters for the NN (Should this be in separate folder?)
    # TODO: parameters for the other NNs?
    parser.add_argument('--ffuncnunit', type=int, default=16)
    parser.add_argument('--percentile', type=float, default=0.1)
    
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--online', action='store_true')
    
    # Visualization
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--saveimage', action='store_true')
    parser.add_argument('--savecmp', action='store_true')
    
    parser.add_argument('--vis', type=str, default='option')

    # Paramter for Diayn
    parser.add_argument('--termprob', type=float, default=0.0)

    args = parser.parse_args()
    return args

def sample_option_trajectories(mdp, args, noptions=None):
    # Sample random trajectories using previously learned options
    # TODO: Sample a trajectories using noptions options.
    
    mdp, state_dim, state_bound, num_actions, action_dim, action_bound = get_mdp_params(args)

    print('action_bound=', action_bound)

    if noptions is None:
        nop = args.noptions
    else:
        nop = noptions

    agent = OptionAgent(sess=None, obs_dim=state_dim, obs_bound=state_bound, num_actions=num_actions, action_dim=action_dim, action_bound=action_bound, num_options=nop+1, init_all=args.initall, high_method='rand', low_method='rand', f_func='rand', epsilon=1.0, batch_size=1, buffer_size=args.snsteps * args.snepisodes, option_batch_size=1, option_buffer_size=args.snsteps * args.snepisodes, name='rand_' + str(nop))
    
    for k in range(1, nop + 1):
        op = CoveringOption(sess=None, experience_buffer=None, obs_dim=state_dim, obs_bound=mdp.bounds(), num_actions=num_actions, action_dim=action_dim, action_bound=action_bound, low_method=args.lowmethod, f_func=args.ffunction, n_units=args.ffuncnunit, init_all=args.initall, reversed_dir=args.reverse, restore=True, name='option' + str(k) + '_' + str(args.ffuncnunit) + '_' + str(args.rseed))

        if args.trajdir == '__default':
            prefix = '.'
        else:
            prefix = args.trajdir
        
        # if args.reverse:
        #     op.restore(prefix + '/vis/' + args.task + 'option' + str(k) + 'rev_' + str(args.ffuncnunit) + '_' + str(args.rseed))
        # else:
        op.restore(prefix + '/vis/' + args.task + 'option' + str(k) + '_' + str(args.ffuncnunit) + '_' + str(args.rseed))
        agent.add_option(op)
        print('restored', k, '-th option')
    
    # buffer_size = args.snsteps * args.snepisodes
    # bfr = ExperienceBuffer(buffer_size=buffer_size)

    # init_x = 0.2
    # init_y = 0.9
    
    for ep in range(1, args.snepisodes + 1):
        # TODO: randomize the initial state?
        mdp.reset()
        # state = mdp.get_init_state()
        # print('state=', state)
        # xy = (12.0, 0.0)
        # xy = (16.0, 12.0)
        # mdp.env.wrapped_env.set_xy(xy)
        # s = mdp.env._get_obs()
        # state = GymState(s, False)
        state = mdp.get_init_state()
        reward = 0
        # print('state=', state)
        # exit
        # state.x = init_x
        # state.y = init_y
        for step in range(1, args.snsteps + 1):
            # print('state=', state)

            if args.saveimage:
                # HACK: To visualize the trajectory I'm adding images.
                data = mdp.env.env._get_image()
                # data = mdp.env.env.clone_full_state()
                # print('data=', data)
            else:
                data = None
                
            action = agent.act(state, reward, train=False, data=data)
            # action = agent.act(state, reward, train=True) # Why was I setting it to true?

            # if reward > 0.0:
            #     print('reward', reward)
            # print('action=', action)

            reward, next_state = mdp.execute_agent_action(action)

            # info = mdp.next_info
            
            # print('info=', info)

            if state.is_terminal():
                # print('reward=', reward)
                # print('term at', next_state)
                next_state = mdp.get_init_state()
                # state.x = 0.2
                # state.y = 0.9
                break

            # trans = [state, action, 0, next_state, 0]
            # bfr.add(trans)

            # print('s =', state)
            # print('a =', action)
            # print('s\'=', next_state)
            # print('isterm?=', next_state.is_terminal())
            state = next_state

    op_bfr = agent.option_buffer
    ex_bfr = agent.experience_buffer

    return op_bfr, ex_bfr

def get_mdp_params(args):
    state_dim = None
    state_bound = None
    num_actions = None
    action_dim = None
    action_bound = None

    # TODO: it is very hard to have a script which contains all
    #       discrete/continuous state/actions.
    #       Should we separete out the tasks, or refactor?
    
    if args.tasktype == 'pinball' or args.tasktype == 'p':
        # TODO: Add parameter for Configuration files by --task argument
        mdp = PinballMDP(cfg=args.task, render=args.render)
        state_dim = 4
        num_actions = len(mdp.get_actions())
        # assert(args.ffunction !=  'fourier')
    elif args.tasktype == 'atari' or args.tasktype == 'atariram':
        grayscale = False
        downscale = True
        # downscale = args.tasktype == 'atari'
        mdp = GymMDP(env_name=args.task, grayscale=grayscale, downscale=downscale, render=args.render)
        # mdp = GymMDP(env_name=args.task, grayscale=True, render=args.render)
        mdp.env.seed(1234)
        state_dims = mdp.env.observation_space.shape
        # print('observation_space=', state_dims)
        if args.tasktype == 'atari':
            state_dim = 1
            for d in state_dims:
                state_dim *= d
            # state_dim = 33600
            # state_dim = 40000 # ?
            if grayscale:
                state_dim = int(state_dim / 3)
            if downscale:
                # state_dim = int(state_dim / 4)
                state_dim = 105 * 80 * 3
        else:
            state_dim = 128
        print('state_dim=', state_dim)
        num_actions = mdp.env.action_space.n

        # TODO: methods are fixed to dqn/ddpg/nn right now.
        print('args.highmethod is overwritten by dqn')
        print('args.lowmethod is overwritten by dqn')
        args.highmethod = 'dqn'
        args.lowmethod = 'dqn'
        # args.ffunction = 'nn'
        assert(args.highmethod == 'dqn')
        assert(args.lowmethod == 'dqn')
        # assert(args.ffunction == 'nn')
    elif args.tasktype == 'mujoco':
        mdp = GymMDP(env_name=args.task, render=args.render)
        mdp.env.seed(1234)
        state_dims = mdp.env.observation_space.shape
        state_dim = 1
        for d in state_dims:
            state_dim *= d
        print('state_dim=', state_dim)

        action_dim = int(mdp.env.action_space.shape[0])
        action_bound = mdp.action_bounds()

        # print(action_dim)
        # Fourier does not work for high dim space.

        # TODO: methods are fixed to dqn/ddpg/nn right now.
        print('args.highmethod is overwritten by dqn')
        print('args.lowmethod is overwritten by ddpg')
        args.highmethod = 'dqn'
        args.lowmethod = 'ddpg'
        # args.ffunction = 'nn'
        assert(args.highmethod == 'dqn')
        assert(args.lowmethod == 'ddpg')
        # assert(args.ffunction == 'nn')
        pass
    elif args.tasktype == 'grid':
        fname = '../tasks/' + args.task
        mdp = make_grid_world_from_file(fname)
        state_dim = 2
        num_actions = 4
    else:
        assert(False)
        pass

    state_bound = mdp.bounds()

    return mdp, state_dim, state_bound, num_actions, action_dim, action_bound

    
