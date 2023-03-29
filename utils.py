# generate train data
from pprint import pprint
def get_train_data(train_path):
    # split  text into sentences for both hidden and observable variables
    # also find all the discrete hidden states
    x = []
    y = []
    with open(train_path, 'r') as file:
        lineCounter = 0 
        for line in file.readlines()[:100]:
            if(len(line.rstrip().split()) == 0):
                continue
            else:
                x.append(line.rstrip().split()[0])
                y.append(line.rstrip().split()[1])
            lineCounter += 1
    return x, y


