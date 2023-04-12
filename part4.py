from nltk.corpus import stopwords
import numpy as np
import sys
import math
import os
import filter_methods as fmt

# preprocess training set - remove stop words
stop_words = set(stopwords.words('english'))
fr_stop_words = set(stopwords.words('french'))

stopwordsdict = {
    "EN":stop_words,
    "FR":fr_stop_words
}

# read training set and remove words that are stopwords
def filter_train_data(lang, stop_words):
    X = []
    y = []
    with open(f'{lang}/train', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if (len(line.rstrip().split()) == 0):
                pass
            elif (line.rstrip().split()[0].lower() in stop_words):
                pass
            else:
                X.append(line.rstrip().split()[0])
                y.append(line.rstrip().split()[1])

    return X, y


def get_test_data(lang):
    test_words = []
    with open(f'{lang}/dev.in', 'r', encoding='utf-8') as f:
        for line in f:
            if len(line.replace("\n", "")) == 0:
                test_words.append("")
            else:
                test_words.append(line.replace("\n", ""))
    return test_words

# generate empty matrix of size (TxO)
def generate_empty_matrix(observed_values, hidden_states):
    return np.zeros((len(hidden_states), len(observed_values)))

def generate_emission_matrix(lang, stop_words):
    # declare k for laplace smoothing
    k = 1

    # get train data
    X, y = fmt.filter_train_data(lang, stopwordsdict)

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
    eps = np.finfo(0.).tiny
    emission_log = np.log(emission_matrix + eps)
    
    # add #UNK# token to every hidden state
    for i in range(len(emission_matrix)):
        emission_matrix[i][-1] = math.log(k/(k+total_count_per_state.get(hidden_states[i])))        
    return emission_log, hidden_states, total_count_per_state, observed_values



# emission probabilities x p(y)
def naive_bayes(emission_matrix, hidden_states: list, total_count_per_state: dict):
    # get probability of states, number of times state appear over number of total appearance of all states
    total_state_appearance = sum(total_count_per_state.values())
    hidden_state_probability = total_count_per_state.copy()

    # get probability of each state, log final value to avoid underflow -> p(y)
    for k,v in hidden_state_probability.items():
        hidden_state_probability[k] = math.log(v/total_state_appearance)

    # add p(y) to emission prob
    for hidden_state, prob in hidden_state_probability.items():
        for i in range(len(emission_matrix[hidden_states.index(hidden_state)])):
            emission_matrix[hidden_states.index(hidden_state)][i] += prob


    return emission_matrix
    


def test(lang, nb_matrix, observed_values: list, hidden_states: list):
    states = []
    state = ""

    words = fmt.get_test_data(lang,stopwordsdict)
    for word in words:
        if len(word) == 0:
            states.append("\n")
        else:
            if word not in observed_values:
                max_unk_prob = float('-inf')
                for i in range(len(nb_matrix)):
                    if nb_matrix[i][-1] > max_unk_prob:
                        max_unk_prob = nb_matrix[i][-1]
                        state = hidden_states[i]
            else:
                max_prob = float('-inf')
                for i in range(len(nb_matrix)):
                    if nb_matrix[i][observed_values.index(word)] > max_prob:
                        max_prob = nb_matrix[i][observed_values.index(
                            word)]
                        state = hidden_states[i]
            states.append(state)

    return states, words, len(words)

def write_to_file(lang, states, words):
    with open(f'{lang}/dev.p4.out', 'w', encoding='utf-8') as f:
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

    emission_log, hidden_states, total_count_per_state, observed_values = generate_emission_matrix(lang, stop_words)
    nb_matrix = naive_bayes(emission_log, hidden_states, total_count_per_state)
    states, words, pred_entities = test(lang, nb_matrix, observed_values, hidden_states)
    write_to_file(lang, states, words)
    os.system(f"python3 ./EvalScript/evalResult.py ./{lang}/dev.out ./{lang}/dev.p4.out")