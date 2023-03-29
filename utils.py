# generate train data
from pprint import pprint
def get_train_data(train_path):
    x = []
    y = []
    y_one = []
    x_one = []
    with open(train_path, 'r') as file:
        lineCounter = 0 
        for line in file.readlines()[:50]:
            if(len(line.rstrip().split()) == 0):
                if lineCounter > 0:
                    y.append(y_one)
                    x.append(x_one)
                    y_one = []
                    x_one = []
            else:
                x_one.append(line.rstrip().split()[0])
                y_one.append(line.rstrip().split()[1])
            lineCounter += 1
    return x, y


x,y = get_train_data("EN/train")
