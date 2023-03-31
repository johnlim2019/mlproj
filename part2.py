from pprint import pprint
import part1 as p1
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
            # print(v_i)
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
    q_u_v = np.multiply(np.log(q_u_v),coeff)
    # print(q_u_v)
    mle_val = np.sum(q_u_v)
    return  q_u_v, mle_val

def get_emission_mle(x:list,y:list,emission_matrix,counts_matrix,observed_values:list,hidden_states:list):
    b_u_o= []
    coeff = []
    for i,u in enumerate(y):
        o = x[i]
        u_i = hidden_states.index(u)
        o_i = observed_values.index(o)
        b_u_o_i = emission_matrix[u_i][o_i]
        b_u_o.append(b_u_o_i)
        coeff.append(counts_matrix[u_i][o_i])
        # if np.isnan(b_u_o):
        #     print(u_i,o_i)
        #     print(u + "->"+ o)
        #     print(type(b_u_o))
        #     print(b_u_o)
    # pprint(b_u_o)
    # print("coefficient")
    # pprint(coeff)
    b_u_o_log = np.multiply(np.log(b_u_o),coeff) # log(0) is inf and when multiplied returns nan
    b_u_o_log = np.nan_to_num(b_u_o_log)# nan to 0
    mle_val = sum(b_u_o_log) 
    # pprint(b_u_o[:100])
    # pprint(b_u_o_log[:100])
    # print(mle_val)
    return  b_u_o_log,mle_val

def get_trans_likelihood_outer(path:str):
    x2,y2 = get_train_data(path)
    y_states = sentence_hidden_states_set(y2)
    count_u_map = count_u(y2,y_states)
    count_u_v_map, train_seq_pairs = count_u_v_1st(y2,y_states)
    transmission_matrix = get_transmission_matrix(count_u_v_map,count_u_map)
    q, mle_trans = get_transmission_mle(train_seq_pairs,transmission_matrix,count_u_v_map)
    return q, mle_trans

def get_em_likelihood_outer(path:str):
    x,y = p1.get_data(path)
    emission_matrix, counts_matrix ,hidden_state_counter, observed_values, hidden_states = p1.generate_emission_matrix(path)
    emls,emle = get_emission_mle(x,y,emission_matrix,counts_matrix,observed_values,hidden_states)
    return emls, emle

def get_log_likelihood_outer(path:str):
    em_q, em_mle = get_em_likelihood_outer(path)
    tr_q, tr_mle = get_trans_likelihood_outer(path)
    return em_mle + tr_mle

if __name__ == '__main__':   
    tr_q, tr_mle = get_trans_likelihood_outer("EN/train")
    print("Part2 Part A: MLE of transmission prob: "+str(tr_mle))

    print(get_log_likelihood_outer("EN/train"))




