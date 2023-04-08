import numpy as np
from part2 import get_transmission_matrix_outer
import os
import sys
from part1 import generate_emission_matrix


def prepare_transmission(t):
    '''
    Append col of zero to left, row of zero to bottom for transmission
    '''
    t = np.concatenate((np.zeros((t.shape[0], 1)), t), axis=1)
    t = np.concatenate((t, np.zeros((1, t.shape[1]))), axis=0)

    return t


def prepare_emissions(e):
    '''
    Append row of zero to bottom, append row of zero with first element 1 to top to emissions
    '''
    fr = np.zeros((1, e.shape[1]))
    fr[0, 0] = 1
    e = np.concatenate((fr, e), axis=0)
    e = np.concatenate((e, np.zeros((1, e.shape[1]))))
    return e


def viterbi(transitions, u, emissions, observations, train_o):
    # Number of states
    K = len(u)
    # Length of observation sequences
    N = len(observations)

    # Initialize probability and backtracking matrices
    V = np.zeros((K, N+2))
    B = np.zeros((K, N+1)).astype(np.int32)

    C = np.array(np.zeros(K))
    C[0] = 1

    # Convert to log
    eps = np.finfo(0.).tiny
    transitions_log = np.log(transitions+eps)
    emissions_log = np.log(emissions+eps)
    C_log = np.log(C+eps)

    V[:, 0] = C_log + emissions_log[:, 0]

    # Fill Viterbi and backtracking matrices
    for i in range(1, N+1):
        for j in range(K):
            t = transitions_log[:, j] + V[:, i-1]

            # Multiply by the emission probability of currently considered node.
            if ((obs := observations[i-1]) in train_o):
                V[j, i] = np.max(t) + emissions_log[j, train_o.index(obs)]
            else:
                V[j, i] = np.max(t) + emissions_log[j, train_o.index('#UNK#')]

            B[j, i-1] = np.argmax(t)


    stp = transitions_log[:, -1] + V[:, -2]

    # Backtracking
    opt_path = np.zeros(N+1).astype(np.int32)
    opt_path[-1] = np.argmax(stp)
    for i in range(N-1, -1, -1):
        opt_path[i] = B[int(opt_path[i+1]), i]

    return opt_path, V, B


def write_results(observations, states, opt_path, file_path='./dev.p2.out'):
    with open(file_path, 'a+', encoding='UTF-8') as f:
        for i, t in enumerate(opt_path[1:]):
            f.write(f'{observations[i]} {states[t]}\n')

        f.write('\n')


def get_observed(test_path):
    '''
    Return a list of list of tweets.
    '''
    with open(test_path, 'r', encoding='UTF-8') as f:
        res = []
        tweet = []
        for line in f:
            line = line.strip()
            if len(line) == 0:
                res.append(tweet)
                tweet = []
            else:
                tweet.append(line)

    return res


def part2(path='EN'):
    test_obs_ls = get_observed(f'{path}/dev.in')
    t_matrix = get_transmission_matrix_outer(f"{path}/train")

    TEST_OUTPUT = f'{path}/dev.p2.out'
    emissions, _, _, observed_values, hidden_states = generate_emission_matrix(
        f"{path}/train")

    u = ['START'] + hidden_states
    v = hidden_states + ['STOP']
    K = len(u)
    transitions = np.zeros((K, K))

    for i in range(K):
        for j in range(K):
            transitions[i][j] = t_matrix[u[i]][v[j]]

    emissions = prepare_emissions(emissions)
    transitions = prepare_transmission(transitions)
    states = hidden_states
    states.insert(0, 'START')
    states.append('STOP')

    if os.path.exists(TEST_OUTPUT):
        os.remove(TEST_OUTPUT)

    # For each tweet, decode.
    for test_obs in test_obs_ls:
        opt_path, dp, backtrack = viterbi(
            transitions, states, emissions, test_obs, observed_values)

        write_results(test_obs, states, opt_path, TEST_OUTPUT)

    return opt_path


if __name__ == "__main__":
    path = sys.argv[1]
    part2(path)
