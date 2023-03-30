# imports
import numpy as np
import utils

def get_data(path):
    X = []
    y = []
    with open(path, 'r') as file:
        for line in file:
            if(len(line.rstrip().split()) == 0):
                pass
            else:
                X.append(line.rstrip().split()[0])
                y.append(line.rstrip().split()[1])
    return X, y

def get_test_data(path):
    test_words = []
    with open(path, 'r') as file:
        for line in file:
            if len(line.replace("\n", "")) == 0:
                pass
            else:
                test_words.append(line.replace("\n", ""))
    return test_words

# generate empty emission matrix of size TxO
def generate_matrix(observed_values, hidden_states):
    return np.zeros((len(hidden_states), len(observed_values)))


def generate_emission_matrix(train_path):

    # get train data
    train_data_X, train_data_y = get_data(train_path)
    
    # get set of observed values, convert to list so can get index
    observed_values = list(set(train_data_X))

    # get set of hidden states, convert to list so can get index
    hidden_states = list(set(train_data_y))
    total_count_per_state = dict.fromkeys(hidden_states, 0)

    # generate emission matrix, storing total counts of each emission from u to o
    emission_matrix = generate_matrix(observed_values, hidden_states)


    # update the counts into the matrix
    for o,u in zip(observed_values, train_data_y):
        total_count_per_state[u] += 1
        emission_matrix[hidden_states.index(u)][observed_values.index(o)] += 1

    # update emission matrix based on formula count(u->o)/count(u), probabilities
    for u in hidden_states:
        for o in observed_values:
            emission_matrix[hidden_states.index(u)][observed_values.index(o)] /= total_count_per_state[u]

    # print(hidden_states)
    # print(observed_values)
    # print(emission_matrix)
    # print(total_count_per_state)
    return emission_matrix, total_count_per_state, observed_values, hidden_states


def test(emission_matrix, test_path, hidden_state_counter:dict, observed_values:list, hidden_states:list):
    tag_dict = {}
    # get test data, and create empty dict
    test_data = list(set(get_test_data(test_path)))
    for word in test_data:
        tag_dict[word] = 0

    for word in test_data:
        if word not in observed_values:
            tag_dict[word] = min(hidden_state_counter, key=hidden_state_counter.get)
        else:
            max_prob = 0
            for i in range(len(emission_matrix)):
                if emission_matrix[i][observed_values.index(word)] > max_prob:
                    max_prob = emission_matrix[i][observed_values.index(word)]
                    tag_dict[word] = hidden_states[i]
    return tag_dict, len(get_test_data(test_path))

def get_f_score(tag_dict, dev_out_path, pred_entities):
    # get dev out data
    dev_out_X, dev_out_y = get_data(dev_out_path)
    predict_correct = 0
    # calculate number of correct predictions
    for i in range(len(dev_out_X)):
        if dev_out_y[i] == tag_dict[dev_out_X[i]]:
            predict_correct += 1

    precision = predict_correct / pred_entities
    recall = predict_correct / len(dev_out_X)
    f_score = 2 / ((1/precision) + (1/recall))
    return f_score

if __name__ == '__main__':
    emission_matrix, hidden_state_counter, observed_values, hidden_states = generate_emission_matrix("EN/train")
    tag_dict, pred_entities = test(emission_matrix, "EN/dev.in", hidden_state_counter, observed_values, hidden_states)
    f_score = get_f_score(tag_dict, "EN/dev.out", pred_entities)
    print(f_score)