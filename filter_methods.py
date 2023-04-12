from nltk.corpus import stopwords
from pprint import pprint
import re 
def filter_train_data(lang,stopwordsdict:dict):
    # common word types are provided one class 
    # stopwords #STOP#
    # punctuation #PUNC# 
    # links and twitter handles #LNK#
    stop = "#STOP#"
    punc = "#PUNC#"
    lnks = "#LNK#"
    X = []
    y = []
    stopwords = stopwordsdict[lang]
    
    with open(f'{lang}/train','r',encoding='utf-8') as f:
        for line in f.readlines():
            line = line.rstrip().split()
            if (len(line)==0):
                pass
            else:
                x = line[0]
                y.append(line[1])
                if (re.search("(?=^http)(.+?)(?=$)",x) != None):
                    x = lnks
                elif (re.search("(?=^)(@\w+?)",x) != None):
                    x = lnks
                # if (re.search("(?=^)(\W+?)(?=$)",x) != None):
                #     x = punc
                # if (x.lower() in stopwords):
                #     x = stop
                x = x.lower()
                X.append(x)
    return  X, y

def get_test_data(lang,stopwordsdict:dict):
    stop = "#STOP#"
    punc = "#PUNC#"
    lnks = "#LNK#"
    X = []
    stopwords = stopwordsdict[lang]

    with open(f'{lang}/dev.in', 'r', encoding='utf-8') as f:
        for line in f:
            if len(line.replace("\n", "")) == 0:
                X.append("")
            else:
                x = line.replace("\n","")
                if (re.search("(?=^http)(.+?)(?=$)",x) != None):
                    x = lnks
                elif (re.search("(?=^)(@\w+?)",x) != None):
                    x = lnks
                # if (re.search("(?=^)(\W+?)(?=$)",x) != None):
                #     x = punc
                # if (x.lower() in stopwords):
                #     x = stop
                x = x.lower()
                X.append(x)
    return X


if __name__ == "__main__":
    stop_words = set(stopwords.words('english'))
    X,y =filter_train_data("EN",stop_words)
    x = get_test_data("EN",stop_words)
    pprint(X)
    pprint(set(y))