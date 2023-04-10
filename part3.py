import os
from pprint import pprint
import math
import pandas as pd
import numpy as np
import part1 as p1
import part2 as p2
import multiprocessing
import pickle


unknown_tag = "#UNK#"
eps = np.finfo(0.0).tiny


def get_train_data(train_path):
    # split  text into sentences for both hidden and observable variables
    # also find all the discrete hidden states
    x = ["START"]
    y = ["START"]
    with open(train_path, "r") as file:
        lineCounter = 0
        for line in file.readlines():
            if len(line.rstrip().split()) == 0:
                if lineCounter > 0:
                    y.append("STOP")
                    y.append("START")
            else:
                x.append(line.rstrip().split()[0])
                y.append(line.rstrip().split()[1])
            lineCounter += 1
        y.pop()
    return x, y


def count_u0_u1_v(y: list, y_states: set):
    # count the instances transition
    # count_u1_u0_v_map[y_i][y_i_1][y_i_2] = the count of the transition
    # the seq_triples each element is [y_i,y_i_1,y_i_2]

    # for START_map it is special map to help us find the first y1 from y0=START.
    # in the main map we include [y0][y1][y2] to measure the probability of seq that reaches y2

    # to include the first START -> y1 transition
    # we create prestart state so we can use the same matrix to include it in the 3d matrix
    seq_triples = []
    count_u1_u0_v_map = {}
    prestart = {"START": {}}
    y_i = y_states.copy()
    y_i.remove("STOP")

    y_i_1 = y_states.copy()
    y_i_1.remove("STOP")
    y_i_1.remove("START")

    y_i_2 = y_states.copy()
    y_i_2.remove("START")
    for i in y_i:
        count_u1_u0_v_map[i] = {}
        for j in y_i_1:
            count_u1_u0_v_map[i][j] = {}
            prestart["START"][j] = eps
            for k in y_i_2:
                count_u1_u0_v_map[i][j][k] = eps

    for i, v_i in enumerate(y):
        if "STOP" in v_i:
            # cannot move to another state from stop
            continue
        if i >= len(y) - 2:
            break  # cos we are reading two places in advance.
        v_i_1 = y[i + 1]
        if "START" in v_i:
            prestart["START"][v_i_1] += 1
        if "STOP" in v_i_1:
            continue
        v_i_2 = y[i + 2]
        seq_triples.append([v_i, v_i_1, v_i_2])
        count_u1_u0_v_map[v_i][v_i_1][v_i_2] += 1

    # add the prestart
    count_u1_u0_v_map["PRESTART"] = prestart
    # pprint(count_u1_u0_v_map)
    # exit()
    return count_u1_u0_v_map, seq_triples


def count_u0_u1(y: list, y_states: set):
    # counts the instances where u0 and u1 pair are the dependent hidden states for a transition change
    # start map is for the counts for the first y1
    start_count_map = {}
    count_u0_u1_map = {}

    # print(y_states)
    start_count_map["START"] = y.count("START")
    y_i = y_states.copy()
    y_i.remove("STOP")
    y_i_1 = y_states.copy()
    y_i_1.remove("START")
    y_i_1.remove("STOP")
    for i in y_i:
        count_u0_u1_map[i] = {}
        for j in y_i_1:
            count_u0_u1_map[i][j] = 0
    # populate

    for i, v_i in enumerate(y):
        if "STOP" in v_i:
            # cannot move to another state from stop
            continue
        if i >= len(y) - 2:
            break
        v_i_1 = y[i + 1]
        if "STOP" in v_i_1:
            continue  # cannot move to another state from stop
        count_u0_u1_map[v_i][v_i_1] += 1

    # add the start and prestart
    count_u0_u1_map["PRESTART"] = start_count_map
    # pprint(count_u0_u1_map)
    # print(y)
    # exit()
    return count_u0_u1_map, start_count_map


def get_transition_matrix_2nd(
    count_u0_u1_v_map: dict,
    count_u0_u1_map: dict,
):
    # count_u1_u0_v_map[y_i][y_i_1][y_i_2] = prob of y_i,y_i_1 -> y_i_2 given y_i,y_i_1
    for u_0 in count_u0_u1_v_map.keys():
        for u_1 in count_u0_u1_v_map[u_0].keys():
            divisor = count_u0_u1_map[u_0][u_1]
            for v in count_u0_u1_v_map[u_0][u_1].keys():
                if divisor == 0:
                    continue
                count_u0_u1_v_map[u_0][u_1][v] /= divisor

    return count_u0_u1_v_map


def get_transition_matrix_2nd_outer(path: str):
    # 2nd order transition log likelihood from training data
    x2, y2 = get_train_data(path)
    # y2 = y2[:50]
    y_states = p2.sentence_hidden_states_set(y2)
    count_u0_u1_map, start_count_map = count_u0_u1(y2, y_states)
    count_u0_u1_v_map, seq_triples = count_u0_u1_v(y2, y_states)
    # pprint(count_u0_u1_v_map)
    trans_matrix = get_transition_matrix_2nd(count_u0_u1_v_map, count_u0_u1_map)
    return trans_matrix, count_u0_u1_map


def get_test_data(path: str):
    # we take the file and split them into their sentences.
    test_words = []
    with open(path, "r") as file:
        sentence = []
        for line in file:
            if len(line.replace("\n", "")) == 0:
                test_words.append(sentence)
                sentence = []
            else:
                sentence.append(line.replace("\n", ""))
    return test_words


def viterbi(transitions, emissions, input_sentence, hidden_state, train_o, emission_count, transition_count):
    k = len(hidden_state)
    N = len(input_sentence) + 2

    y_predict_all = []

    # create vertice matrix with pandas
    hidden_state_ext = hidden_state.copy()
    hidden_state_ext.extend(["START", "STOP", "PRESTART"])
    state_combis = set()
    for u in hidden_state_ext:
        for u1 in hidden_state_ext:
            if u == "PRESTART" and u1 != "START":
                continue
            if u != "PRESTART" and u1 == "START":
                continue
            if u == "START" and u1 == "STOP":
                continue
            if u1 == "PRESTART":
                continue
            state_combis.add((u, u1))
    state_combis = list(state_combis)
    V = pd.DataFrame(index=pd.MultiIndex.from_tuples(state_combis), columns=np.arange(N)).fillna(-math.inf)  # vertice map
    # each vertice cell is the p of the combination
    B = pd.DataFrame(index=pd.MultiIndex.from_tuples(state_combis), columns=np.arange(N))  # backtracker

    # set start
    V.loc[[("PRESTART", "START")], 0] = 1
    # print(V.to_string())

    # forward pass
    for i in range(1, N - 1):  # iterate over the remaining visible vars excluding STOP
        word_index = i - 1
        x = input_sentence[word_index]
        if x not in train_o:
            x = "#UNK#"
        for v in hidden_state:  # curr state
            for u1 in hidden_state_ext:
                for u0 in hidden_state_ext:
                    try:
                        emissions_p = np.log(emissions[hidden_state.index(v), train_o.index(x)]) * emission_count[hidden_state.index(v), train_o.index(x)]
                    except Exception as e:
                        print(hidden_state)
                        print(e)
                        print(v)
                        print(u0)
                        print(u1)
                        exit()
                    try:
                        p = transitions[u0][u1][v]
                    except:
                        # often this means that it is an impossible transition
                        continue
                    transition_p = np.log(p) * transition_count[u0][u1]

                    p = V.loc[[(u0, u1)], i - 1] + emissions_p + transition_p
                    p = np.float64(p)
                    if (p > V.loc[[(u1, v)], i]).bool():
                        V.loc[[(u1, v)], i] = p
                        B.loc[[(u1, v)], i] = u0
                        # print(V.loc[[(u0, u1)], i])
                        # print(p)
                        # print(u0)
                        # print(i)
    # add stop here we only use the transition prob
    j = N - 1
    v = "STOP"
    for u1 in hidden_state:
        for u0 in hidden_state:
            try:
                p = transitions[u0][u1][v]
            except:
                # often this means that it is an impossible transition
                print("F")
                continue
            transition_p = np.log(p) * transition_count[u0][u1]
            p = V.loc[[(u0, u1)], j - 1] * transition_p
            # print(V.loc[[(u1,v)],j])
            p = np.float64(p)
            if (p > V.loc[[(u1, v)], j]).bool():
                V.loc[[(u1, v)], j] = p
                B.loc[[(u1, v)], j] = u0
    V.to_csv("v.csv")
    B.to_csv("b.csv")
    print(input_sentence)
    # backtracking
    # we go back states each iteration,
    # each cell is the second back spot
    y_predict = []
    curr_u1_v = V[N - 1].idxmax()
    for i in range(N - 1, 0, -1):
        next_state = B.loc[curr_u1_v][i]
        if isinstance(next_state, str):
            y_predict.insert(0, curr_u1_v[1])
            curr_u1_v = (next_state, curr_u1_v[0])
        else:
            # cannot reach START
            for j in range(int(i)):
                y_predict.insert(0, ("0"))
            break
    # print(y_predict)
    y_predict = y_predict[:-1]
    print(y_predict)
    # print(input_sentence)
    y_predict_all.append(y_predict)
    return y_predict_all


def viterbi_loop(transition_matrix_dict, emission_matrix, sentences, hidden_states, observed_values, count_u_o_matrix, count_u0_u1_matrix, output):
    for s in sentences:
        out_i = viterbi(
            transition_matrix_dict,
            emission_matrix,
            s,
            hidden_states,
            observed_values,
            count_u_o_matrix,
            count_u0_u1_matrix,
        )
        output.extend(out_i)
    return output

def split_list(sentences):
    size = math.ceil(len(sentences) / 6)
    print(size) 
    lastsize= len(sentences) - 5 * size
    print(lastsize)
    s = []
    while sentences:
        chunk, sentences = sentences[:size], sentences[size:]
        s.append(chunk)
        # print(chunk)
    return s[0],s[1],s[2],s[3],s[4],s[5]


def predict_file(trainingPath: str, testPath: str):
    sentences = get_test_data(testPath)
    (
        transition_matrix_dict,
        count_u0_u1_matrix,
    ) = get_transition_matrix_2nd_outer(trainingPath)
    (
        emission_matrix,
        count_u_o_matrix,
        hidden_state_counter,
        observed_values,
        hidden_states,
    ) = p1.generate_emission_matrix(trainingPath)

    # add min value to emissions so we can log
    emission_matrix += eps

    print("hidden states " + str(hidden_states))
    print(len(hidden_states))
    # sentences = [
    #     "holy shit ! are you kidding ! shitstorm".split(),
    #     ["hi"],
    #     "hi there".split(),
    #     "last night was epic dude omg".split(),
    #     "that was terrble and shit".split(),
    #     "kfjsl slkdfjsk omg what is this".split(),
    # ]

    N = len(observed_values) - 1  # because one of them is #UNK#
    print(len(sentences))
    all_predicts = []
    s1,s2,s3,s4,s5,s6 = split_list(sentences)
    out1 = multiprocessing.Manager().list()
    out2 = multiprocessing.Manager().list()
    out3 = multiprocessing.Manager().list()
    out4 = multiprocessing.Manager().list()
    out5 = multiprocessing.Manager().list()
    out6 = multiprocessing.Manager().list()
    t1 = multiprocessing.Process(
        target=viterbi_loop,
        args=(transition_matrix_dict, emission_matrix, s1, hidden_states, observed_values, count_u_o_matrix, count_u0_u1_matrix, out1),
    )
    t2 = multiprocessing.Process(
        target=viterbi_loop,
        args=(transition_matrix_dict, emission_matrix, s2, hidden_states, observed_values, count_u_o_matrix, count_u0_u1_matrix, out2),
    )
    t3 = multiprocessing.Process(
        target=viterbi_loop,
        args=(transition_matrix_dict, emission_matrix, s3, hidden_states, observed_values, count_u_o_matrix, count_u0_u1_matrix, out3),
    )
    t4 = multiprocessing.Process(
        target=viterbi_loop,
        args=(transition_matrix_dict, emission_matrix, s4, hidden_states, observed_values, count_u_o_matrix, count_u0_u1_matrix, out4),
    )
    t5 = multiprocessing.Process(
        target=viterbi_loop,
        args=(transition_matrix_dict, emission_matrix, s5, hidden_states, observed_values, count_u_o_matrix, count_u0_u1_matrix, out5),
    )
    t6 = multiprocessing.Process(
        target=viterbi_loop,
        args=(transition_matrix_dict, emission_matrix, s6, hidden_states, observed_values, count_u_o_matrix, count_u0_u1_matrix, out6),
    )
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t5.start()
    t6.start()
    print("STARTED ALL PROCESSES")
    t2.join()
    t1.join()
    t3.join()
    t4.join()
    t5.join()
    t6.join()
    t1.terminate()
    t2.terminate()
    t3.terminate()
    t4.terminate()
    t5.terminate()
    t6.terminate()
    print(out1)
    all_predicts = list(out1) + list(out2) + list(out3) + list(out4) + list(out5) + list(out6)
    pprint(len(all_predicts))
    return all_predicts


def compile_dev_out(x_testdata: list, y_predictions: list, folder: str):
    # x_testdata this takes in the data from part3's get_test_data method
    # this is is nested list, list of a sentence list
    # y_predictions are the output from the y_predict_file
    # this is a nested list, list of sentence list
    all_sentence_string = ""
    for ind, x_sentence in enumerate(x_testdata):
        # print(ind)
        # print(x_sentence)
        y_sentence = y_predictions[ind]
        # print(y_sentence)
        # y_sentence.remove("STOP")
        # y_sentence.remove("START")
        sentence_string = ""
        if ind > 0:
            sentence_string += "\n"
        for j, o_j in enumerate(x_sentence):
            y_j = y_sentence[j]
            sentence_string += o_j + " " + y_j + "\n"
        all_sentence_string += sentence_string

    with open(folder + "/dev.p3.out", "w") as f:
        f.write(all_sentence_string)
    return


def check_results(folder):
    os.system("python3 ./EvalScript/evalResult.py ./" + folder + "/dev.out ./" + folder + "/dev.p3.out")
    return


if __name__ == "__main__":
    y_predict_all = predict_file("EN/train", "EN/dev.in")
    with open("./EN_y.pkl",'wb') as f:
        pickle.dump(y_predict_all,f)
    with open("./EN_y.pkl",'rb') as f:
        y_predict_all = pickle.load(f)
    x_test = get_test_data("EN/dev.in")
    print(len(y_predict_all))
    print(len(y_predict_all) - len(x_test))
    compile_dev_out(x_test, y_predict_all, "EN")
    check_results("EN")

    y_predict_all = predict_file("FR/train", "FR/dev.in")
    with open("./FR_y.pkl",'wb') as f:
        pickle.dump(y_predict_all,f)
    with open("./FR_y.pkl",'rb') as f:
        y_predict_all = pickle.load(f)
    x_test = get_test_data("FR/dev.in")
    print(len(x_test))
    # print(x_test[0])
    # print(y_predict_all[0])
    # print(len(y_predict_all[0]) - len(x_test[0]))
    compile_dev_out(x_test, y_predict_all, "FR")
    check_results("FR")
