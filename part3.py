import math
import random
from pprint import pprint

import numpy as np
import part1 as p1
import part2 as p2

unknown_x_prob = 0.001


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


def predict_file(trainingPath: str, testPath: str):
    sentences = get_test_data(testPath)
    transmission_matrix, count_u0_u1_matrix, start_map, start_count_map = get_transmission_matrix_2nd_outer(
        trainingPath
    )
    (
        emission_matrix,
        count_u_o_matrix,
        hidden_state_counter,
        observed_values,
        hidden_states
    ) = p1.generate_emission_matrix(trainingPath)

    # sentences = [
    #     ["hi"],
    #     "hi there".split(),
    #     "last night was epic dude omg".split(),
    #     "that was terrble and shit".split(),
    #     "kfjsl slkdfjsk omg what is this".split()
    # ]

    log_likelihood = 0
    y_predict_all = []
    for x in sentences:
        # print(x)
        # set up start
        # because we have no previous two, we make a non deterministic guess for y1
        # based on the probability distribution of the y1 given y0
        # unlike the rest, we are using pairs and triples.

        # settling transmission from START
        v_ls = list(start_map["START"].keys())
        v_p = []
        for u1 in start_map["START"].keys():
            v_p.append(start_map["START"][u1])

        # pprint(v_ls)
        # we do not want to just pick the best option
        # as technically all states have a change to be the first state
        # so we choose according to their probability
        v = np.random.choice(v_ls, p=v_p)
        # max = 0
        # indmax = 0
        # for ind,value in enumerate(v_p):
        #     if value > max:
        #         max = value
        #         indmax = ind
        # v = v_ls[indmax]
        a = v_p[v_ls.index(v)]
        # print(v)
        # print(a)
        count_u0 = start_count_map["START"]

        if a == 0:
            # if transmission is not known its p is small
            # one out of all the possible outputs from START
            a = 1 / len(v_ls)

        # settling emissions
        o1 = x[0]
        o1_known = True
        if o1 not in observed_values:
            o1_known = False
        if o1_known:
            b1 = emission_matrix[hidden_states.index(
                v)][observed_values.index(o1)]
            count_u1_o1 = count_u_o_matrix[hidden_states.index(
                v)][observed_values.index(o1)]
        if o1_known == False or b1 == 0:
            # unknown emission, means the p is v small.
            # it has only occurred once count only 1
            # if the emission has not occured before this is the first time.
            b1 = 1 / (len(observed_values) + 1)
            count_u1_o1 = 1

        p = count_u0 * np.log(a) + count_u1_o1 * np.log(b1)

        # set up the sentence predidction
        y_predict = ["START", v]
        # print(p)
        log_likelihood += p

        # iterating over the observable states
        # for y1 to yn i.e guess the y2 given o2, y0, y1
        completed_x = []
        for index, x_1 in enumerate(x):
            completed_x.append(x_1)
            # we have the guessed the first hidden state already. we start at o2
            # to do so we skip the first observation states

            if index == 0:
                continue
            high_p = -math.inf

            # get observable element
            o2 = x[index]
            o2_known = True
            # check if emissions is known
            if o2 not in observed_values:
                o2_known = False

            # get the y1,y2(u0,u1)
            # print(y_predict)
            u0 = y_predict[-2]
            u1 = y_predict[-1]
            # print()
            # print(u1)
            # get the list possible v/y2
            possible_states = transmission_matrix[u0][u1]
            # print(possible_states)
            # from within the sentence iteration we cannot reach STOP, as they do not have observable states
            # START is not one of the possible v states, it is not in this list alr
            v_list = list(possible_states.keys())
            v_list.remove("STOP")
            for v in v_list:
                if o2_known:
                    b3 = emission_matrix[hidden_states.index(v)][
                        observed_values.index(o2)
                    ]
                    count_v_o = count_u_o_matrix[hidden_states.index(v)][
                        observed_values.index(o2)
                    ]
                if o2_known == False or b3 == 0:
                    # unknown emission, means the p is v small.
                    # it has only occurred once count only 1
                    b3 = 1 / (len(observed_values) + 1)
                    count_v_o = 1
                a = transmission_matrix[u0][u1][v]
                count_u0 = count_u0_u1_matrix[u0][u1]
                if a == 0:
                    # if transmission is not known its p is small
                    # one out of all the possible combinations of u0, u1
                    a = 1 / len(hidden_states) / len(hidden_states)
                if count_u0 == 0:
                    # this is the first time we have observed it in respect to the training data aset
                    count_u0 = 1
                p = count_u0 * np.log(a) + count_v_o * np.log(b3)
                if p >= high_p:
                    high_p = p
                    highest_key = v
            # add the hidden state and p
            log_likelihood += high_p
            y_predict.append(highest_key)

        # exit sentence inner loop
        # from the yn to stop there is only one possible log likelihood value.
        # given yn-1, yn -> yn+1
        # we only measure the last emission prob in the calculation,
        u0 = y_predict[-2]
        u1 = y_predict[-1]
        v = "STOP"
        o1 = x[-1]
        o1_known = True
        # check if emissions is known
        if o1 not in observed_values:
            o1_known = False
        if o1_known:
            b1 = emission_matrix[hidden_states.index(
                u1)][observed_values.index(o1)]
            count_u1_o1 = count_u_o_matrix[hidden_states.index(u1)][
                observed_values.index(o1)
            ]
        if o1_known == False or b1 == 0:
            # unknown emission, means the p is v small.
            # it has only occurred once count only 1
            b1 = 1 / (len(observed_values) + 1)
            count_u1_o1 = 1

        # find the transmission probabiltiy
        if u0 == "START":
            # this case is START y0 STOP
            # if the first element is START
            # in the transmission matrix START is only in pairs : START, U_1
            # so adjust the way we request a value
            a = start_map[u0][u1]
            count_u0 = start_count_map[u0]
        else:
            a = transmission_matrix[u0][u1]["STOP"]
            count_u0 = count_u0_u1_matrix[u0][u1]
        if a == 0:
            # if transmission is not known its p is small
            # one out of all the possible y states
            # as technically all states have a chance of ending sentence
            a = 1 / len(hidden_states)
        if count_u0 == 0:
            # it only occurred once.
            count_u0 = 1

        # log likelihood.
        # print(a)
        p = count_u0 * np.log(a) + count_u1_o1 * np.log(b1)
        y_predict.append("STOP")
        # add the log likelihood and record the sentence y values
        log_likelihood += p
        y_predict_all.append(y_predict)
        print(y_predict)
        print(x)

        # we completed one sentence, move to the next sentence.
        # exit()
    # print(log_likelihood)
    return log_likelihood, y_predict_all


def compile_dev_out(x_testdata: list, y_predictions: list, folder: str):
    # x_testdata this takes in the data from part3's get_test_data method
    # this is is nested list, list of a sentence list
    # y_predictions are the output from the y_predict_file
    # this is a nested list, list of sentence list
    all_sentence_string = ""
    for ind, x_sentence in enumerate(x_testdata):
        y_sentence = y_predictions[ind]
        y_sentence.remove("STOP")
        y_sentence.remove("START")
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


# def get_transmission_mle_2nd(triples:list,trans_matrix:dict,count_u0_u1_v_map:dict):
#     # this returns the transmission log likelihood
#     # sum of log(prob of u0,u1->v given u0,u1) * count(u0,u1)
#     a_u0_u1_v = []
#     coeff = []
#     for i in triples:
#         coeff.append(count_u0_u1_v_map[i[0]][i[1]][i[2]])
#         # if trans_matrix[i[0]][i[1]][i[2]] > 1:
#             # print(i)
#             # print(trans_matrix[i[0]][i[1]][i[2]])
#         a_u0_u1_v.append(trans_matrix[i[0]][i[1]][i[2]])
#     a_u0_u1_v = np.multiply(np.log(a_u0_u1_v),coeff)
#     # pprint(a_u0_u1_v)
#     mle_val = np.sum(a_u0_u1_v)
#     return a_u0_u1_v, mle_val

# def get_log_likelihood_2nd(path:str):
#     # joint log likelihood for transmission and emission
#     em_q, em_mle = p2.get_em_likelihood_outer(path)

#     # 2nd order transmission log likelihood
#     x2,y2 = p2.get_train_data(path)
#     y_states = p2.sentence_hidden_states_set(y2)
#     count_u0_u1_map = count_u0_u1(y2,y_states)
#     count_u0_u1_v_map,seq_triples = count_u0_u1_v(y2,y_states)
#     # pprint(count_u0_u1_v_map)
#     trans_matrix = get_transmission_matrix_2nd(count_u0_u1_v_map,count_u0_u1_map)
#     transls, transmle = get_transmission_mle_2nd(seq_triples,trans_matrix,count_u0_u1_v_map)
#     return transmle + em_mle


if __name__ == "__main__":
    loglikelihood, y_predict_all = predict_file("EN/train", "EN/dev.in")
    x_test = get_test_data("EN/dev.in")
    # print(x_test[0])
    # print(y_predict_all[0])
    # print(len(y_predict_all[0]) - len(x_test[0]))
    compile_dev_out(x_test, y_predict_all, "EN")

    loglikelihood, y_predict_all = predict_file("FR/train", "FR/dev.in")
    x_test = get_test_data("FR/dev.in")
    # print(x_test[0])
    # print(y_predict_all[0])
    # print(len(y_predict_all[0]) - len(x_test[0]))
    compile_dev_out(x_test, y_predict_all, "FR")
