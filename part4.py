import numpy as np
import os
import sys
import re
def get_data(path):
    stop = "#STOP#"
    punc = "#PUNC#"
    lnks = "#LNK#"
    X = []
    y = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.rstrip().split()
            if (len(line)==0):
                pass
            else:
                x = line[0]
                x = x.lower()
                y.append(line[1])
                if (re.search("(?=^http)(.+?)(?=$)",x) != None):
                    x = lnks
                elif (re.search("(?=^)(@\w+?)",x) != None):
                    x = lnks
                # elif (re.search("(?=^)(\W+?)(?=$)",x) != None):
                #     x = punc
                X.append(x)
    return X, y

def get_observed(test_path):
    '''
        Return a list of list of tweets.
        reading testing data not training data
    '''
    stop = "#STOP#"
    punc = "#PUNC#"
    lnks = "#LNK#"
    with open(test_path, 'r', encoding='UTF-8') as f:
        res = []
        tweet = []
        for line in f:
            line = line.strip().lower()
            if (re.search("(?=^http)(.+?)(?=$)",line) != None):
                line = lnks
            elif (re.search("(?=^)(@\w+?)",line) != None):
                line = lnks
            # elif (re.search("(?=^)(\W+?)(?=$)",line) != None):
            #     line = punc               
            if len(line) == 0:
                res.append(tweet)
                tweet = []
            else:
                tweet.append(line)

    return res


def generate_matrix(observed_values, hidden_states):
    return np.zeros((len(hidden_states), len(observed_values)))


def generate_emission_matrix(train_path):
    k = 1

    # get train data
    train_data_X, train_data_y = get_data(train_path)

    # get set of observed values, convert to list so can get index
    observed_values = list(set(train_data_X))
    observed_values.sort()
    observed_values.append("#UNK#")

    # get set of hidden states, convert to list so can get index
    hidden_states = list(set(train_data_y))
    hidden_states.sort()
    total_count_per_state = dict.fromkeys(hidden_states, 0)

    # generate emission matrix, storing total counts of each emission from u to o
    emission_matrix = generate_matrix(observed_values, hidden_states)
    count_u_o_matrix = generate_matrix(observed_values, hidden_states)

    # update the counts into the matrix
    for o, u in zip(train_data_X, train_data_y):
        total_count_per_state[u] += 1
        count_u_o_matrix[hidden_states.index(u)][observed_values.index(o)] += 1

    # update emission matrix based on formula count(u->o)/count(u), probabilities
    emission_matrix = count_u_o_matrix.copy()
    for u in hidden_states:
        for o in observed_values:
            emission_matrix[hidden_states.index(
                u)][observed_values.index(o)] /= total_count_per_state[u]

    # add UNK token to every state
    for i in range(len(emission_matrix)):
        emission_matrix[i][-1] = k / \
            (k+total_count_per_state.get(hidden_states[i]))

    return emission_matrix, count_u_o_matrix, total_count_per_state, observed_values, hidden_states



def get_train_data(train_path):
    # split  text into sentences for both hidden and observable variables
    # also find all the discrete hidden states
    # this is used for hidden values, we can ignore the x values.
    x = ['START']
    y = ['START']
    with open(train_path, 'r') as file:
        lineCounter = 0
        for line in file.readlines():
            if (len(line.rstrip().split()) == 0):
                if lineCounter > 0:
                    y.append("STOP")
                    y.append("START")
                    x.append("STOP")
                    x.append("START")
            else:
                x.append(line.rstrip().split()[0].lower())
                y.append(line.rstrip().split()[1])
            lineCounter += 1
    return x, y


def sentence_hidden_states_set(y) -> set:
    return set(y)


def count_u(y, states_y) -> dict:
    # return dict of the number of each state in sentence
    count_u_map = {}
    for state in states_y:
        count_u_map[state] = y.count(state)
    return count_u_map


def count_u_v_1st(y: list, y_states: set):
    # count the instances of U->V in first order and return the count in a dict
    # format:
    # count_u_v_map[y_i][y_i+1]
    # set up the map
    # y_i cannot include stop we cannot move to another state from stop
    # y_i+1 cannot include start we cannot move from any state to start
    seq_pairs = []
    count_u_v_map = {}
    # print(y_states)
    y_i = y_states.copy()
    y_i.remove("STOP")
    y_i_1 = y_states.copy()
    y_i_1.remove("START")
    for i in y_i:
        count_u_v_map[i] = {}
        for j in y_i_1:
            count_u_v_map[i][j] = 0
    # pprint(count_u_v_map)
    # count the state changes
    for i, v_i in enumerate(y):
        if "STOP" in v_i:
            # cannot move to another state from stop
            continue
        if i >= len(y)-1:
            # print(v_i)
            break
        v_i_1 = y[i+1]
        seq_pairs.append([v_i, v_i_1])
        count_u_v_map[v_i][v_i_1] += 1
    # pprint(count_u_v_map)
    return count_u_v_map, seq_pairs


def get_transmission_matrix(count_u_v_map: dict, count_u_map: dict):
    # return the matrix containing the probability of u->v given u
    # it returns a dict map[u][v] = prob of u->v given u
    for u_i in count_u_v_map.keys():
        divisor = count_u_map[u_i]
        for v_i in count_u_v_map[u_i].keys():
            count_u_v_map[u_i][v_i] /= divisor
    # pprint(count_u_v_map)
    return count_u_v_map


def get_transmission_matrix_outer(path: str):
    # returns the transmission probability matrix based on training data
    x2, y2 = get_train_data(path)
    y_states = sentence_hidden_states_set(y2)
    count_u_map = count_u(y2, y_states)
    count_u_v_map, train_seq_pairs = count_u_v_1st(y2, y_states)
    transmission_matrix = get_transmission_matrix(count_u_v_map, count_u_map)
    return transmission_matrix


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


def write_results(observations, states, opt_path, file_path='./dev.p4.out'):
    with open(file_path, 'a+', encoding='UTF-8') as f:
        for i, t in enumerate(opt_path[1:]):
            f.write(f'{observations[i]} {states[t]}\n')

        f.write('\n')


def get_observed_full(test_path):
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
    # print(test_obs_ls[:25])
    test_obs_full = get_observed_full(f'{path}/dev.in')
    t_matrix = get_transmission_matrix_outer(f"{path}/train")

    TEST_OUTPUT = f'{path}/dev.p4.out'
    emissions, _, _, observed_values, hidden_states = generate_emission_matrix(
        f"{path}/train")
    # print(observed_values)

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
    for index,test_obs in enumerate(test_obs_ls):
        opt_path, dp, backtrack = viterbi(
            transitions, states, emissions, test_obs, observed_values)

        write_results(test_obs_full[index], states, opt_path, TEST_OUTPUT)

    return opt_path


if __name__ == '__main__':
    try:
        lang = sys.argv[1]
    except:
        sys.exit("Please provide a language path as an argument (python part4.py <lang_path>). Possible values are 'EN' and 'FR' (without quotes)")
    part2(lang)
    # os.system(f"python3 ./EvalScript/evalResult.py ./{lang}/dev.out ./{lang}/dev.p4.out")