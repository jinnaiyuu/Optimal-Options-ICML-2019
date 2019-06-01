import numpy as np
import random
from time import sleep

from simple_rl.tasks import GridWorldMDP
from simple_rl.tasks import GymMDP # Gym

# from simple_rl.planning.ValueIterationClass import ValueIteration
from options.option_generation.ValueIterationClass import ValueIteration

def get_h(mdp):
    return hash(str(mdp.env.unwrapped.ale.getRAM()))

def GetAdjacencyMatrix(mdp):
    # print('mdp type=', type(mdp))
    vi = ValueIteration(mdp) # TODO: the VI class does sampling which doesn't work for stochastic planning.
    vi.run_vi()
    A, states = vi.compute_adjacency_matrix()

    for k in range(A.shape[0]):
        A[k][k] = 0

    intToS = dict()
    for i, s in enumerate(states):
        intToS[i] = s
    return A, intToS # (matrix, dict)

def GetIncidenceMatrix(mdp, n_traj=1, eps_len=10):
    '''
    Sample transitions and build amn incidence matrix.
    Returns: A: incidence matrix
             states: mapping from matrix index to state
    '''
    # TODO: What is the best way to represent the incidence matrix?
    # Required output: 


    pairs = [] # List of state transition pairs (s, s')
    hash_to_ind = dict() # Dictionary of state -> index
    ind_to_s = dict()

    actions = mdp.get_actions()
    cur_s = mdp.get_init_state()

    # if type(mdp) is GymMDP:
    #     cur_h = get_h(mdp)
    # else:
    cur_h = hash(cur_s)
    hash_to_ind[cur_h] = 0
    ind_to_s[0] = cur_s
    

    n_states = 1

    # Sample transitions
    for i in range(n_traj):
        mdp.reset()
        cur_s = mdp.get_init_state()
        # if type(mdp) is GymMDP:
        #     cur_h = get_h(mdp)
        # else:
        cur_h = hash(cur_s)
            
        # print('cur_s=', cur_s)
        # print('type(cur_s)=', type(cur_s))
        
        for j in range(eps_len):
            # sleep(0.01)
            # print('hash=', hash(cur_s.data.tostring()))
            # TODO: Sample trajectory based on a particular policy rather than a random walk?
            a = random.choice(actions)
            _, next_s = mdp.execute_agent_action(a)
            # TODO: The GymMDP is not returning the correct observation???

            # obs = mdp.env.unwrapped.ale.getRAM()
            # print('obs=', obs)

            # print('next_h=', get_h(next_s))

            # print('nextstate=', mdp.next_state.data)
            # print('env.state=', mdp.env.state)
            
            # print('type(next_s)=', type(next_s))

            # if type(mdp) is GymMDP:
            #     next_h = get_h(mdp)
            # else:
            next_h = hash(next_s)

            if next_h in hash_to_ind.keys():
                next_i = hash_to_ind[next_h]
            else:
                # print('next_s=', next_s)
                next_i = n_states
                hash_to_ind[next_h] = next_i
                ind_to_s[next_i] = next_s
                n_states += 1

            p = (hash_to_ind[cur_h], next_i)
            pairs.append(p)

            cur_s = next_s
            cur_h = next_h

    # print('pairs=', pairs)
    
    matrix = np.zeros((n_states, n_states), dtype=int)

    for i in range(len(pairs)):
        if pairs[i][0] is not pairs[i][1]:
            matrix[pairs[i][0], pairs[i][1]] = 1

    print('n_states=', n_states)
    print('matrix.shape=', matrix.shape)
    print('matrix=', matrix)

    mdp.reset()
    return matrix, ind_to_s

