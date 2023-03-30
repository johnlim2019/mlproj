import numpy as np 
from pprint import pprint
import part2 as p2

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
    count_u0_u1_map = {}
    # print(y_states)
    y_i = y_states.copy()
    y_i.remove("STOP")
    y_i_1 = y_states.copy()
    y_i_1.remove("START")
    y_i_1.remove("STOP")
    for i in y_i:
        count_u0_u1_map[i] = {}
        for j in y_i_1:
            count_u0_u1_map[i][j] = 0;
    for i,v_i in enumerate(y):
        if "STOP" in v_i:
            # cannot move to another state from stop
            continue
        if i >= len(y) - 2:
            break
        v_i_1 = y[i+1]
        if "STOP" in v_i_1:
            continue # cannot move to another state from stop
        count_u0_u1_map[v_i][v_i_1] += 1
        # pprint(count_u0_u1_map)
    return count_u0_u1_map        

def get_transmission_matrix_2nd(count_u0_u1_v_map:dict,count_u0_u1_map:dict):
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

def get_transmission_mle_2nd(triples:list,trans_matrix:dict,count_u0_u1_v_map:dict):
    a_u0_u1_v = []
    coeff = []
    for i in triples:
        coeff.append(count_u0_u1_v_map[i[0]][i[1]][i[2]])
        # if trans_matrix[i[0]][i[1]][i[2]] > 1:
            # print(i)
            # print(trans_matrix[i[0]][i[1]][i[2]])
        a_u0_u1_v.append(trans_matrix[i[0]][i[1]][i[2]])
    a_u0_u1_v = np.multiply(-1*np.log(a_u0_u1_v),coeff)
    # pprint(a_u0_u1_v)
    mle_val = np.sum(a_u0_u1_v)
    return a_u0_u1_v, mle_val        

if __name__ == '__main__':
    x,y = p2.get_train_data('EN/train')
    y_states = p2.sentence_hidden_states_set(y)
    count_u0_u1_map = count_u0_u1(y,y_states)
    count_u0_u1_v_map,seq_triples = count_u0_u1_v(y,y_states)
    # pprint(count_u0_u1_v_map)
    trans_matrix = get_transmission_matrix_2nd(count_u0_u1_v_map,count_u0_u1_map)
    ls, mle_val = get_transmission_mle_2nd(seq_triples,trans_matrix,count_u0_u1_v_map)
    print(mle_val)