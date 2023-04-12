# imports
import numpy as np
import sys
import os


def get_data(path):
    X = []
    y = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            if (len(line.rstrip().split()) == 0):
                pass
            else:
                X.append(line.rstrip().split()[0])
                y.append(line.rstrip().split()[1])
    return X, y


def get_test_data(path):
    test_words = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            if len(line.replace("\n", "")) == 0:
                test_words.append("")
            else:
                test_words.append(line.replace("\n", ""))
    return test_words

# generate empty emission matrix of size TxO


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


def test(emission_matrix, test_path, observed_values: list, hidden_states: list):
    states = []
    state = ""

    # get test words
    words = get_test_data(test_path)

    for word in words:
        if len(word) == 0:
            states.append("\n")
        else:
            if word not in observed_values:
                max_unk_prob = 0
                for i in range(len(emission_matrix)):
                    if emission_matrix[i][-1] > max_unk_prob:
                        max_unk_prob = emission_matrix[i][-1]
                        state = hidden_states[i]
            else:
                max_prob = 0
                for i in range(len(emission_matrix)):
                    if emission_matrix[i][observed_values.index(word)] > max_prob:
                        max_prob = emission_matrix[i][observed_values.index(
                            word)]
                        state = hidden_states[i]
            states.append(state)
    return states, words, len(words)


def write_to_file(path, states, words):
    with open(path, 'w') as f:
        for i in range(len(words)):
            if states[i] != "\n":
                f.write(words[i] + ' ' + states[i] + "\n")
            else:
                f.write(states[i])


if __name__ == '__main__':
    try:
        lang = sys.argv[1]
    except:
        sys.exit("Please provide a language path as an argument (python part1.py <lang_path>). Possible values are 'EN' and 'FR' (without quotes)")

    emission_matrix, count_u_o_matrix, hidden_state_counter, observed_values, hidden_states = generate_emission_matrix(
        f"{lang}/train")
    states, words, pred_entities = test(
        emission_matrix, f"{lang}/dev.in", observed_values, hidden_states)
    write_to_file(f"{lang}/dev.p1.out", states, words)
    os.system(f"python3 ./EvalScript/evalResult.py ./{lang}/dev.out ./{lang}/dev.p1.out")