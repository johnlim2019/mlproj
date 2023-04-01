import math
import numpy as np
import random 
from pprint import pprint
import part2 as p2
import part1 as p1

unknown_x_prob = 0.001

def count_u0_u1_v(y:list,y_states:set):
    # count the instances transmission 
    # count_u1_u0_v_map[y_i][y_i_1][y_i_2] = the count of the transmission
    # the seq_triples each element is [y_i,y_i_1,y_i_2]
    seq_triples = []
    count_u1_u0_v_map = {}
    y_i = y_states.copy()
    y_i.remove("STOP")
    
    y_i_1 = y_states.copy()
    y_i_1.remove("START")
    y_i_1.remove("STOP")

    y_i_2 = y_states.copy()
    y_i_2.remove("START")
    for i in y_i:
        count_u1_u0_v_map[i] = {}
        for j in y_i_1:
            count_u1_u0_v_map[i][j] = {}
            for k in y_i_2:
                count_u1_u0_v_map[i][j][k] = 0
    for i,v_i in enumerate(y):
        if "STOP" in v_i:
            # cannot move to another state from stop
            continue
        if i >= len(y) - 2:
            break
        v_i_1 = y[i+1]
        if "STOP" in v_i_1:
            continue
        v_i_2 = y[i+2]
        seq_triples.append([v_i,v_i_1,v_i_2])
        # pprint(v_i + v_i_1 + v_i_2)
        count_u1_u0_v_map[v_i][v_i_1][v_i_2] += 1
    # pprint(count_u1_u0_v_map)
    return count_u1_u0_v_map, seq_triples

def count_u0_u1(y:list,y_states:set):
    # counts the instances where u0 and u1 pair are the dependent hidden states for a transmission change
    count_u0_u1_map = {}
    # print(y_states)
    num_of_start = y.count("START")
    y_i = y_states.copy()
    y_i.remove("STOP")
    y_i_1 = y_states.copy()
    y_i_1.remove("START")
    y_i_1.remove("STOP")
    for i in y_i:
        count_u0_u1_map[i] = {}
        for j in y_i_1:
            count_u0_u1_map[i][j] = 0
    for i,v_i in enumerate(y):
        if "STOP" in v_i:
            # cannot move to another state from stop
            continue
        if i >= len(y) - 2:
            break
        v_i_1 = y[i+1]
        if "STOP" in v_i_1:
            continue # cannot move to another state from stop
        if "START" in v_i:
            # START we consider do not consider the pair unique, 
            # we count all number of starts
            count_u0_u1_map[v_i][v_i_1] = num_of_start
        else:
            count_u0_u1_map[v_i][v_i_1] += 1
        # pprint(count_u0_u1_map)
    return count_u0_u1_map        

def get_transmission_matrix_2nd(count_u0_u1_v_map:dict,count_u0_u1_map:dict):
    # count_u1_u0_v_map[y_i][y_i_1][y_i_2] = prob of y_i,y_i_1 -> y_i_2 given y_i,y_i_1
    for u_0 in count_u0_u1_v_map.keys():
        for u_1 in count_u0_u1_v_map[u_0].keys():
            divisor = count_u0_u1_map[u_0][u_1]
            for v in count_u0_u1_v_map[u_0][u_1].keys():
                numerator = count_u0_u1_v_map[u_0][u_1][v]
                if divisor == 0 and numerator == 0:
                    continue
                count_u0_u1_v_map[u_0][u_1][v] /= divisor
                # if v == "STOP":
                #     print(divisor)
    return count_u0_u1_v_map

def get_transmission_matrix_2nd_outer(path:str):
    # 2nd order transmission log likelihood from training data
    x2,y2 = p2.get_train_data(path)
    y_states = p2.sentence_hidden_states_set(y2)
    count_u0_u1_map = count_u0_u1(y2,y_states)
    count_u0_u1_v_map,seq_triples = count_u0_u1_v(y2,y_states)
    # pprint(count_u0_u1_v_map)
    trans_matrix = get_transmission_matrix_2nd(count_u0_u1_v_map,count_u0_u1_map)
    return trans_matrix, count_u0_u1_map

def get_test_data(path:str):
    # we take the file and split them into their sentences. 
    test_words = []
    with open(path, 'r') as file:
        sentence = []
        for line in file:
            if len(line.replace("\n", "")) == 0:
                test_words.append(sentence)
                sentence=[]
            else:
                sentence.append(line.replace("\n", "").lower())
    return test_words


def predict_file(trainingPath:str,testPath:str):
    sentences =  get_test_data(testPath)
    transmission_matrix, count_u0_u1_matrix = get_transmission_matrix_2nd_outer(trainingPath)    
    emission_matrix, count_u_o_matrix, hidden_state_counter, observed_values, hidden_states = p1.generate_emission_matrix(trainingPath)

    log_likelihood = 0
    for x in sentences[:1]:
        # set up start 
        # pprint(transmission_matrix["START"])
        print(x)
        high_p = -math.inf
        highest_key_start_u1 = None 
        highest_key_start_v = None
        o1 = x[0]
        o2 = x[1]
        o1_known = o2_known = True
        if o1 not in observed_values:
            o1_known = False
        if o2 not in observed_values:
            o2_known = False 
        for u1 in transmission_matrix["START"].keys():
            subdir = transmission_matrix["START"][u1]
            # from start we cannot reach STOP
            # START is not one of the possible v states, it is not in this list alr
            v_list = list(subdir.keys())
            v_list.remove("STOP")
            for v in v_list:
                if o1_known:
                    b3 = emission_matrix[hidden_states.index(u1)][observed_values.index(o1)] 
                    count_u1_o1 = count_u_o_matrix[hidden_states.index(u1)][observed_values.index(o1)]
                if o1_known == False or b3 == 0:
                    # unknown emission, means the p is v small. 
                    # it has only occurred once count only 1
                    b3 = 1 /(len(observed_values)+1)
                    count_u1_o1 = 1
                if o2_known:
                    b2 = emission_matrix[hidden_states.index(v)][observed_values.index(o2)]
                    count_v_o2 = count_u_o_matrix[hidden_states.index(v)][observed_values.index(o2)]        
                if o2_known == False or b2 ==0 :
                    b2 = 1 /(len(observed_values)+1)
                    count_v_o2 = 1                    
                a = transmission_matrix["START"][u1][v]                                                
                count_u0_u1 = count_u0_u1_matrix["START"][u1]
                if a == 0:
                    # if transmission is not known its p is small
                    # one out of all the possible cells the transmission matrix 
                    a = 1/len(hidden_states)/len(hidden_states)/len(hidden_states)
                if count_u0_u1 == 0:
                    # it only occurred once
                    count_u0_u1 = 1                                                
                p = (count_u0_u1*np.log(a)+count_u1_o1*np.log(b3)+count_v_o2*np.log(b2))
                # print(p)
                if p >= high_p:
                    high_p = p
                    highest_key_start_u1 = u1
                    highest_key_start_v = v
        print(high_p)
        y_predict = ["START",highest_key_start_u1,highest_key_start_v]   
        print(y_predict)
        log_likelihood += high_p

        # iterating over the observable states
        # for y1 to yn i.e guess the y3 given o3, y1, y2
        completed_x =[]
        for index,x_1 in enumerate(x):
            completed_x.append(x_1)
            # we have the guessed the first two hidden states already. we start at o3
            if index == 0 or index == 1:
                continue
            # once we reach the second last observable element stop, 
            # the iteration is looking one x value forward 
            # the last one and stop triple should be tested outside this loop. (special case)
            if index >= len(x) - 1:
                break

            high_p = -math.inf
            highest_key_start_v = None
            
            # get observable element
            o3 = x[index]
            o3_known = True
            # check if emissions is known 
            if o3 not in observed_values:
                o3_known = False
        
            # get the y1,y2(u0,u1)
            # print(y_predict)
            u0 = y_predict[-2]
            u1 = y_predict[-1]
            # print(u0)
            # print(u1)
            # get the list possible v/y3
            subdir = transmission_matrix[u0][u1]
            # print(subdir)
            # from within the sentence iteration we cannot reach STOP, as they do not have observable states
            # START is not one of the possible v states, it is not in this list alr
            v_list = list(subdir.keys())
            v_list.remove("STOP")
            for v in v_list:
                if o3_known:
                    b3 = emission_matrix[hidden_states.index(v)][observed_values.index(o3)] 
                    count_v_o = count_u_o_matrix[hidden_states.index(v)][observed_values.index(o3)]
                if o3_known == False or b3 == 0:
                    # unknown emission, means the p is v small. 
                    # it has only occurred once count only 1          
                    b3 = 1 /(len(observed_values)+1)
                    count_v_o = 1                
                a = transmission_matrix[u0][u1][v]                                                
                count_u0_u1 = count_u0_u1_matrix[u0][u1]
                if a == 0:
                    # if transmission is not known its p is small
                    # one out of all the possible cells the transmission matrix 
                    a = 1/len(hidden_states)/len(hidden_states)/len(hidden_states)
                if count_u0_u1 == 0:
                    # this is the first time we have observed it in respect to the training data aset
                    count_u0_u1 = 1                                                      
                p = (count_u0_u1*np.log(a)+count_v_o*np.log(b3))
                if p >= high_p:
                    high_p = p
                    highest_key_start_v = v
            # add the hidden state and p 
            log_likelihood += high_p
            y_predict.append(v)
        # exit sentence inner loop
        # from the yn to stop there is only one possible log likelihood value. 
        # given yn-1, yn,  yn+1
        u0 = y_predict[-2]
        u1 = y_predict[-1]
        v = "STOP"
        o0 = x[-2]
        o1 = x[-1]
        o0_known = True
        # check if emissions is known 
        if o0 not in observed_values:
            o0_known = False
        o1_known = True
        # check if emissions is known 
        if o1 not in observed_values:
            o1_known = False
        
        if o0_known:
            b0 = emission_matrix[hidden_states.index(u0)][observed_values.index(o0)] 
            count_u0_o0 = count_u_o_matrix[hidden_states.index(u0)][observed_values.index(o0)]
        if o0_known == False or b0 == 0:
            # unknown emission, means the p is v small. 
            # it has only occurred once count only 1
            b0 = 1 /(len(observed_values)+1)
            count_u0_o0 = 1
        if o1_known:
            b1 = emission_matrix[hidden_states.index(u1)][observed_values.index(o1)]
            count_u1_o1 = count_u_o_matrix[hidden_states.index(u1)][observed_values.index(o1)]        
        if o1_known == False or b1 ==0 :
            b1 = 1 /(len(observed_values)+1)
            count_u1_o1 = 1                    
        a = transmission_matrix["START"][u1][v]                                                
        count_u0_u1 = count_u0_u1_matrix["START"][u1]
        if a == 0:
            # if transmission is not known its p is small
            # one out of all the possible cells the transmission matrix 
            a = 1/len(hidden_states)/len(hidden_states)/len(hidden_states)
        if count_u0_u1 == 0:
            # it only occurred once
            count_u0_u1 = 1                                                
        p = (count_u0_u1*np.log(a)+count_u1_o1*np.log(b3)+count_v_o2*np.log(b2))




    print(len(y_predict)-len(x))
    print(completed_x)
    print(y_predict)
    print(log_likelihood)


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

if __name__ == '__main__':    
    predict_file("EN/train","EN/dev.in")