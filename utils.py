# generate train data
from pprint import pprint
def get_train_data(train_path):
    # split  text into sentences for both hidden and observable variables
    # also find all the discrete hidden states
    x = ['START']
    y = ['START']
    with open(train_path, 'r') as file:
        lineCounter = 0 
        for line in file.readlines()[:100]:
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


