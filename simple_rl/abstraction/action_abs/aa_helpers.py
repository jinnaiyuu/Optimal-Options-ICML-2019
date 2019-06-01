from copy import copy


# Other imports.
from simple_rl.planning.ValueIterationClass import ValueIteration
from simple_rl.tasks import GridWorldMDP
from simple_rl.abstraction.action_abs.PredicateClass import Predicate
from simple_rl.abstraction.action_abs.InListPredicateClass import InListPredicate
from simple_rl.abstraction.action_abs.OptionClass import Option
from simple_rl.abstraction.action_abs.PolicyFromDictClass import PolicyFromDict
from simple_rl.abstraction.action_abs.IntrinsitcMDP import IntrinsicMDP

from simple_rl.run_experiments import run_single_agent_on_mdp
from simple_rl.agents import QLearningAgent, LinearQAgent

#
from simple_rl.abstraction.action_abs.ClassifierPredicate import ClassifierPredicate

# ------------------------
# -- Goal Based Options --
# ------------------------
def make_goal_based_options(mdp_distr):
    '''
    Args:
        mdp_distr (MDPDistribution)

    Returns:
        (list): Contains Option instances.
    '''

    goal_list = set([])
    for mdp in mdp_distr.get_all_mdps():
        vi = ValueIteration(mdp)
        state_space = vi.get_states()
        for s in state_space:
            if s.is_terminal():
                goal_list.add(s)

    options = set([])
    for mdp in mdp_distr.get_all_mdps():

        init_predicate = Predicate(func=lambda x: True)
        term_predicate = InListPredicate(ls=goal_list)
        o = Option(init_predicate=init_predicate,
                    term_predicate=term_predicate,
                    policy=_make_mini_mdp_option_policy(mdp),
                    term_prob=0.0)
        options.add(o)

    return options


def make_subgoal_options(mdp, goal_list, init_space=None, vectors=None, n_trajs=100, n_steps=100, classifier='list', policy='vi'):
    '''
    Args:
        mdp
        goal_list: set of lists.
        init_space: list of states.

    Returns:
        (list): Contains Option instances.
    '''

    if classifier == 'list':
        init_predicate = InListPredicate(ls=init_space)
    elif classifier == 'svc':
        init_predicate = ClassifierPredicate(init_space)
    else:
        print('Error: unknown predicate for init condition:', classifier)
        assert(False)

    options = set([])
    # print('init_space=', init_space)
    for i, gs in enumerate(goal_list):

        # print('goals=', g)
        # print('type(g)=', g)
        # init_predicate = Predicate(func=lambda x: True)
        # init_predicate = InListPredicate(ls=init_space)
        
        ############################
        # Termination set is set to (the subgoal state) + (unknown region).
        ############################
        term = copy(init_space)

        # print('term=', term, type(term))
        # print('type(term)=', type(term))
        # print('gs=', gs)
        for g in gs:
            # print('g=', g, type(g))
            if g in term:
                term.remove(g)
        
        if classifier == 'list':
            term_predicate = InListPredicate(ls=term, true_if_in=False)
        elif classifier == 'svc':
            term_predicate = ClassifierPredicate(term, true_if_in=False)
        else:
            print('Error: unknown predicate for init condition:', classifier)
            assert(False)


        if policy == 'vi':
            vector = dict()
            for g in gs:
                vector[hash(g)] = 1
            mdp_ = IntrinsicMDP(intrinsic_reward=vector, mdp=mdp)
            o = Option(init_predicate=init_predicate,
                       term_predicate=term_predicate,
                       policy=_make_mini_mdp_option_policy(mdp_, n_iters=100),
                       term_prob=0.0)
        elif policy == 'dqn':
            o = Option(init_predicate=init_predicate,
                       term_predicate=term_predicate,
                       policy=_make_dqn_option_policy(mdp, vectors[i], n_trajs=n_trajs, n_steps=n_steps),
                       term_prob=0.0)
        else:
            print('Error: unknown policy for options:', policy)
            assert(False)

            
                    # policy=_make_mini_mdp_option_policy(mdp),
        options.add(o)

    return options

def make_point_options(mdp, pairs, policy='vi'):
    '''
    Args:
        mdp
        pairs: a list of pairs. Each pair is a list containing init set and term set.

    Returns:
        (list): Contains Option instances.
    '''

    options = set([])
    for pair in pairs:
        init = pair[0]
        term = pair[1]
        if type(init) is not list:
            init = [init]
        if type(term) is not list:
            term = [term]
        # init_predicate = Predicate(func=lambda x: True)
        init_predicate = InListPredicate(ls=init)
        term_predicate = InListPredicate(ls=term)

        if policy == 'vi':
            o = Option(init_predicate=init_predicate,
                       term_predicate=term_predicate,
                       policy=_make_mini_mdp_option_policy(mdp, n_iters=100),
                       term_prob=0.0)
        elif policy == 'dqn':
            o = Option(init_predicate=init_predicate,
                       term_predicate=term_predicate,
                       policy=_make_dqn_option_policy(mdp, term[0]),
                       term_prob=0.0)
        else:
            assert(False)
        options.add(o)

    return options


def _make_dqn_option_policy(mdp, subgoal, n_trajs=100, n_steps=100):
    '''
    LEARN-DQN-AGENT-ON-MDP
    '''

    # TODO: mdp + subgoal
    # TODO: How much should we train each policy?
    #       Near optimal for now?

    env_name = mdp.env_name
    in_mdp = IntrinsicMDP(subgoal, env_name)


    # Build a subgoal reward function
    # TODO: implement a reward function based on the eigenvector
    # def intrinsic_r(state):
    #     if state in subgoal:
    #         return 1.0
    #     else:
    #         return 0.0

    # print('type(subgoal)=', subgoal)
    
    
    num_feats = in_mdp.get_num_state_feats()
    dqn_agent = LinearQAgent(in_mdp.get_actions(), num_feats)

    run_single_agent_on_mdp(dqn_agent, in_mdp, episodes=n_trajs, steps=n_steps)


    return dqn_agent.policy


def _make_mini_mdp_option_policy(mini_mdp, n_iters=1000):
    '''
    Args:
        mini_mdp (MDP)

    Returns:
        Policy
    '''
    # Solve the MDP defined by the terminal abstract state.
    mini_mdp_vi = ValueIteration(mini_mdp, delta=0.001, max_iterations=n_iters, sample_rate=10)
    iters, val = mini_mdp_vi.run_vi()

    o_policy_dict = make_dict_from_lambda(mini_mdp_vi.policy, mini_mdp_vi.get_states())
    o_policy = PolicyFromDict(o_policy_dict)

    # print('type(policy)=', type(o_policy.get_action))

    return o_policy.get_action

def make_dict_from_lambda(policy_func, state_list):
    policy_dict = {}
    for s in state_list:
        policy_dict[s] = policy_func(s)

    return policy_dict
