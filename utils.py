# generate train data
def get_train_data(train_path):
    X = []
    y = []
    with open(train_path, 'r') as file:
        for line in file:
            if(len(line.rstrip().split()) == 0):
                pass
            else:
                X.append(line.rstrip().split()[0])
                y.append(line.rstrip().split()[1])
    return X, y