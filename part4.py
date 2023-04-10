import nltk
from nltk.corpus import stopwords
import numpy as np
import os
import sys
import math

# preprocess training set - remove stop words
stop_words = set(stopwords.words('english'))


# read training set and remove words that are stopwords
def filter_train_data(path, stop_words):
    X = []
    y = []
    with open(f'{path}/train', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if (len(line.rstrip().split()) == 0):
                pass
            elif (line.rstrip().split()[0].lower() in stop_words):
                pass
            else:
                X.append(line.rstrip().split()[0])
                y.append(line.rstrip().split()[1])

    return X, y


# generate empty matrix of size (TxO)
def generate_empty_matrix(observed_values, hidden_states):
    return np.zeros((len(hidden_states), len(observed_values)))

def generate_emission_matrix(path):
    # declare k for laplace smoothing
    k = 1

    # get train data
    X, y = filter_train_data(path)

    # get set of observed values, convert to list so can index
    observed_values = list(set(X))
    observed_values.sort()
    # add #UNK# token for probability of words not seen in training set
    observed_values.append("#UNK#")

    # get set of hidden states, convert to list so can index
    hidden_states = list(set(y))
    hidden_states.sort()
    total_count_per_state = dict.fromkeys(hidden_states, 0)

    # generate emission matrix, storing total counts of each emission from u to o
    emission_matrix = generate_empty_matrix(observed_values, hidden_states)

    # update counts into matrix
    for o,u in zip(X, y):
        total_count_per_state[u] += 1
        emission_matrix[hidden_states.index(u)][observed_values.index(o)] += 1

    # update emission matrix based on formula count(u->o)/count(u), apply log to avoid underflow
    for u in hidden_states:
        for o in observed_values:
            emission_matrix[hidden_states.index(u)][observed_values] = math.log((emission_matrix[hidden_states.index(u)][observed_values]) / (total_count_per_state[u]))

    # add #UNK# token to every hidden state
    for i in range(len(emission_matrix)):
        emission_matrix[i][-1] = math.log(k/(k+total_count_per_state.get(hidden_states[i])))        
    return emission_matrix, hidden_states, total_count_per_state, observed_values



# emission probabilities x p(y)
def naive_bayes(emission_matrix, hidden_states: list, total_count_per_state: dict, observed_values: list):
    # get probability of states, number of times state appear over number of total appearance of all states
    total_state_appearance = sum(total_count_per_state.values())
    hidden_state_probability = total_count_per_state.copy()

    # get probability of each state, log final value to avoid underflow -> p(y)
    for k,v in hidden_state_probability:
        hidden_state_probability[k] = math.log(v/total_state_appearance)

    # add p(y) to emission prob
    


if __name__ == '__main__':
    path = sys.argv[1]


    X, y = filter_train_data(path,  stop_words)
    print(len(X), len(y))
    print(X[0:15])