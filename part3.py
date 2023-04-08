import os
from pprint import pprint

import numpy as np
import part1 as p1
import part2 as p2

unknown_tag = "#UNK#"


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
    # count the instances transmission
    # count_u1_u0_v_map[y_i][y_i_1][y_i_2] = the count of the transmission
    # the seq_triples each element is [y_i,y_i_1,y_i_2]

    # for START_map it is special map to help us find the first y1 from y0=START.
    # in the main map we include [y0][y1][y2] to measure the probability of seq that reaches y2
    seq_triples = []
    count_u1_u0_v_map = {}
    start_map = {}
    y_i = y_states.copy()
    y_i.remove("STOP")

    y_i_1 = y_states.copy()
    y_i_1.remove("START")
    y_i_1.remove("STOP")

    y_i_2 = y_states.copy()
    y_i_2.remove("START")
    for i in y_i:
        if i == "START":
            # for predicting the first y1 from start
            start_map[i] = {}
        count_u1_u0_v_map[i] = {}
        for j in y_i_1:
            if i == "START":
                start_map[i][j] = 0
            count_u1_u0_v_map[i][j] = {}
            for k in y_i_2:
                count_u1_u0_v_map[i][j][k] = 0
    for i, v_i in enumerate(y):
        if "START" in v_i:
            # special case for START we only look one place in advanced.
            # but we also want the START y1 y2 for transmmission for y2
            v_i_1 = y[i + 1]
            seq_triples.append([v_i, v_i_1])
            start_map[v_i][v_i_1] += 1
        if "STOP" in v_i:
            # cannot move to another state from stop
            continue
        if i >= len(y) - 2:
            break  # cos we are reading two places in advance.
        v_i_1 = y[i + 1]
        if "STOP" in v_i_1:
            continue
        v_i_2 = y[i + 2]
        seq_triples.append([v_i, v_i_1, v_i_2])
        count_u1_u0_v_map[v_i][v_i_1][v_i_2] += 1
    # pprint(count_u1_u0_v_map)
    return count_u1_u0_v_map, seq_triples, start_map


def count_u0_u1(y: list, y_states: set):
    # counts the instances where u0 and u1 pair are the dependent hidden states for a transmission change
    # start map is for the counts for the first y1
    start_count_map = {}
    count_u0_u1_map = {}

    # print(y_states)
    y_i = y_states.copy()
    y_i.remove("STOP")
    y_i_1 = y_states.copy()
    y_i_1.remove("START")
    y_i_1.remove("STOP")
    for i in y_i:
        if "START" in i:
            start_count_map[i] = 0
            # this is a special case we only count number of start
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
        if "START" in v_i:
            # START we are only guessing for the first v_i values, one state in advance.
            # we count all number of starts
            start_count_map[v_i] += 1
        else:
            count_u0_u1_map[v_i][v_i_1] += 1
        # pprint(count_u0_u1_map)
    return count_u0_u1_map, start_count_map


def get_transmission_matrix_2nd(
    count_u0_u1_v_map: dict,
    count_u0_u1_map: dict,
    start_map: dict,
    start_count_map: dict,
):
    # count_u1_u0_v_map[y_i][y_i_1][y_i_2] = prob of y_i,y_i_1 -> y_i_2 given y_i,y_i_1
    for u_0 in count_u0_u1_v_map.keys():
        for u_1 in count_u0_u1_v_map[u_0].keys():
            divisor = count_u0_u1_map[u_0][u_1]
            for v in count_u0_u1_v_map[u_0][u_1].keys():
                if divisor == 0:
                    continue
                count_u0_u1_v_map[u_0][u_1][v] /= divisor
    for start in start_map.keys():
        divisor = start_count_map[start]
        for v in start_map[start].keys():
            start_map[start][v] /= divisor
    return count_u0_u1_v_map, start_map


def get_transmission_matrix_2nd_outer(path: str):
    # 2nd order transmission log likelihood from training data
    x2, y2 = get_train_data(path)
    # y2 = y2[:50]
    y_states = p2.sentence_hidden_states_set(y2)
    count_u0_u1_map, start_count_map = count_u0_u1(y2, y_states)
    count_u0_u1_v_map, seq_triples, start_map = count_u0_u1_v(y2, y_states)
    # pprint(count_u0_u1_v_map)
    trans_matrix, start_map = get_transmission_matrix_2nd(
        count_u0_u1_v_map, count_u0_u1_map, start_map, start_count_map
    )
    return trans_matrix, count_u0_u1_map, start_map, start_count_map


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
                sentence.append(line.replace("\n", "").lower())
    return test_words


def viterbi(transitions, emissions, input_sentence, hidden_state, train_o, emission_count, transmission_count):
    k = len(hidden_state)
    N = len(input_sentence)

    # convert to log the transitions and emissions
    eps = np.finfo(0.).tiny
    translations_log = np.log(transitions+eps)
    emissions_log = np.log(emissions+eps)

    # init vertice and backtrack
    V = np.zeros((k, k, N+2))
    B = np.zeros((k, k, N+1))  # from y0 to yn

    # now we add to the vertice the START -> y1
    # From start it influences only one possible yi
    # START is index 0 for the i
    start = np.zeros((k, k))
    start[:, 0] = 1
    start_log = np.log(start+eps)
    V[:, :, 0] = start_log + emissions_log[:, 0]
    # print(V.shape)

    # run the algorithm
    # i,j,k
    # y1,y2 -> y3
    for i_seq in range(1, N+1):  # we already got x1
        for i in range(k):  # y1
            for j in range(k):  # y2
                # this i are the transitions of i,j to the all possible jk
                start_state = hidden_state.copy()
                start_state.append("START")
                t_coeff = transmission_count[start_state[i]][hidden_state[j]]
                t = translations_log[i, j, :] * t_coeff + V[i, j, i_seq-1]  # plus the previous
                if ((obs := input_sentence[i_seq-1]) in train_o):
                    temp = np.max(t) + emissions_log[i, train_o.index(obs)] * emission_count[i][train_o.index(obs)]
                else:
                    temp = np.max(t) + emissions_log[i, train_o.index("#UNK#")] * emission_count[i][train_o.index("#UNK#")]
                if ((obs := input_sentence[i_seq-2]) in train_o):
                    V[i, j, i_seq] = temp + emissions_log[j, train_o.index(obs)] * emission_count[i][train_o.index(obs)]
                else:
                    V[i, j, i_seq] = temp + emissions_log[j, train_o.index("#UNK#")] * emission_count[i][train_o.index("#UNK#")]
                # record index of the hidden state with highest t
                B[i, j, i_seq-1] = np.argmax(t)

    # add STOP
    for i in range(k):
        for j in range(k):
            t = translations_log[i, j, :] + V[i, j, -1]
            V[i, j, -1] = np.max(t)
            B[i, j, -1] = np.argmax(t)

    # print("hidden states " + str(hidden_state))
    # print("sentence "+str(input_sentence))
    # print(V)
    # print(B)
    # backtracking over the B matrix
    predict_y = np.zeros(N)
    predict_y[-1] = np.argmax(V[:,:,-1])
    for seq_i in range(N-3,-1,-1):
        y3 = int(predict_y[seq_i+2])
        y2 = int(predict_y[seq_i+1])
        try:
            predict_y[seq_i] = B[y2,y3,seq_i]
        except Exception as e:
            print(type(e).__name__+": "+e)
            print(input_sentence)
            print(predict_y)
            print(y3)
            print(y2)
            exit()
        
    predict_y_str = []
    for i in predict_y:
        predict_y_str.insert(0,hidden_state[int(i)-1])
    # print()
    # print(input_sentence)
    # print(predict_y_str)
    # print(predict_y)
    return predict_y_str, V, B

def predict_file(trainingPath: str, testPath: str):
    sentences = get_test_data(testPath)
    (
        transmission_matrix_dict,
        count_u0_u1_matrix,
        start_map,
        start_count_map
    ) = get_transmission_matrix_2nd_outer(trainingPath)
    (
        emission_matrix,
        count_u_o_matrix,
        hidden_state_counter,
        observed_values,
        hidden_states
    ) = p1.generate_emission_matrix(trainingPath)

    print("hidden states "+str(hidden_states))
    # sentences = [
    #     ["hi"],
    #     "hi there".split(),
    #     "last night was epic dude omg".split(),
    #     "that was terrble and shit".split(),
    #     "kfjsl slkdfjsk omg what is this".split()
    # ]

    k = len(hidden_states)
    N = len(observed_values) - 1  # because one of them is #UNK#

    # this is K+1 x K x K+1
    # transition matrix in numpy array
    # the additional rows are for the START and STOP
    ith_hidden_state = hidden_states.copy()
    ith_hidden_state.insert(0, "START")
    jkth_hidden_state = hidden_states.copy()
    jkth_hidden_state.append("STOP")
    transmission_matrix = np.zeros((k+1, k, k+1))
    for i in range(k+1):
        for j in range(k):
            for jk in range(k+1):
                transmission_matrix[i][j][jk] = transmission_matrix_dict[ith_hidden_state[i]][hidden_states[j]][jkth_hidden_state[jk]]
    # for the start
    # For start we have to use a separate matrix as it is only 1 dimension
    # start to y1
    start_y1 = np.zeros(k)
    for i in range(k):
        start_y1[i] = start_map["START"][hidden_states[i]]

    # now we have the emission and transition p matrices
    # we call viterbi for each sentence
    all_output = []
    for x in sentences:
        output, V, B =viterbi(transmission_matrix,emission_matrix,x,hidden_states,observed_values, count_u_o_matrix, count_u0_u1_matrix)
        all_output.append(output)
    
    return all_output


def compile_dev_out(x_testdata: list, y_predictions: list, folder: str):
    # x_testdata this takes in the data from part3's get_test_data method
    # this is is nested list, list of a sentence list
    # y_predictions are the output from the y_predict_file
    # this is a nested list, list of sentence list
    all_sentence_string = ""
    for ind, x_sentence in enumerate(x_testdata):
        y_sentence = y_predictions[ind]
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
    os.system("python3 ./EvalScript/evalResult.py ./"+folder+"/dev.out ./"+folder+"/dev.p3.out")
    return

if __name__ == "__main__":
    y_predict_all = predict_file("EN/train", "EN/dev.in")
    # exit()
    x_test = get_test_data("EN/dev.in")
    # print(x_test[0])
    # print(y_predict_all[0])
    # print(len(y_predict_all[0]) - len(x_test[0]))
    compile_dev_out(x_test, y_predict_all, "EN")
    check_results("EN")

    y_predict_all = predict_file("FR/train", "FR/dev.in")
    x_test = get_test_data("FR/dev.in")
    # print(x_test[0])
    # print(y_predict_all[0])
    # print(len(y_predict_all[0]) - len(x_test[0]))
    compile_dev_out(x_test, y_predict_all, "FR")
    check_results("FR")