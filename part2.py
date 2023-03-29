from pprint import pprint
import utils
import numpy as np



def get_train_data(train_path):
    # split  text into sentences for both hidden and observable variables
    # also find all the discrete hidden states
    x = ['START']
    y = ['START']
    with open(train_path, 'r') as file:
        lineCounter = 0 
        for line in file.readlines():
            if(len(line.rstrip().split()) == 0):
                if lineCounter > 0:
                    y.append("STOP")
                    y.append("START")
                    x.append("STOP")
                    x.append("START")
            else:
                x.append(line.rstrip().split()[0])
                y.append(line.rstrip().split()[1])
            lineCounter += 1
    return x, y

def sentence_hidden_states_set(y)->set:    
    return set(y)

def count_u(y,states_y)->dict:
    # return dict of the number of each state in sentence
    count_u_map = {}
    for state in states_y:        
        count_u_map[state] = y.count(state)
    return count_u_map

def count_y_x_1st(y:list,x:list):
    seq_pairs = []
    count_u_v_map = {}
    return count_u_v_map, seq_pairs


def count_u_v_1st(y:list,y_states:set):
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
            count_u_v_map[i][j] = 0;
    # pprint(count_u_v_map)
    # count the state changes
    for i,v_i in enumerate(y):
        if "STOP" in v_i:
            # cannot move to another state from stop
            continue
        if i >= len(y)-1:
            print(v_i)
            break
        v_i_1 = y[i+1]        
        seq_pairs.append([v_i,v_i_1])
        count_u_v_map[v_i][v_i_1] += 1
    # pprint(count_u_v_map)
    return count_u_v_map, seq_pairs

def get_transmission_matrix(count_u_v_map:dict,count_u_map:dict):
    for u_i in count_u_v_map.keys():
        divisor = count_u_map[u_i]
        for v_i in count_u_v_map[u_i].keys():
            count_u_v_map[u_i][v_i] /=divisor
    # pprint(count_u_v_map)
    return count_u_v_map

def get_transmission_mle(pairs:list,trans_matrix:dict,count_u_v_map:dict):
    q_u_v = []
    coeff = []
    for i in pairs:
        coeff.append(count_u_v_map[i[0]][i[1]])
        q_u_v.append(trans_matrix[i[0]][i[1]])
    # print(q_u_v)
    q_u_v = np.multiply(-1*np.log(q_u_v),coeff)
    print(q_u_v)
    mle_val = np.sum(q_u_v)
    return  q_u_v, mle_val

x,y = get_train_data("EN/train")
y_states = sentence_hidden_states_set(y)
count_u_map = count_u(y,y_states)
count_u_v_map, train_seq_pairs = count_u_v_1st(y,y_states)
transmission_matrix = get_transmission_matrix(count_u_v_map,count_u_map)
q, mle_trans = get_transmission_mle(train_seq_pairs,transmission_matrix,count_u_v_map)
print("Part2 Part A: MLE of transmission prob: "+str(mle_trans))



