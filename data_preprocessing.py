
import numpy as np

def vocabs():
    with open('vocab.csv','r') as f:
        vocab=f.readlines()
    vocab=[word.strip('\n') for word in vocab]
    vocab.append("END")
    vocab=sorted(vocab)
    char_index = dict((c, i) for i, c in enumerate(vocab))
    return len(vocab),char_index
    
def set_flag(i,len_vocab):
    tmp = np.zeros(len_vocab)
    tmp[i] = 1
    return list(tmp)

# Truncate names and create the matrix
def prepare_X(X):
    maxlen=16
    len_vocab,char_index=vocabs()
    new_list = []
    trunc_train_name = [str(i)[0:maxlen] for i in X]
    
    for i in trunc_train_name:
        tmp = [set_flag(char_index[j],len_vocab) for j in str(i)]
        for k in range(0,maxlen - len(str(i))):
            tmp.append(set_flag(char_index["END"],len_vocab))
        new_list.append(tmp)

    return new_list

def prepare_y(y):
    new_list = []
    for i in y:
        if i == 0:
            new_list.append([1,0])
        else:
            new_list.append([0,1])

    return new_list


    
    