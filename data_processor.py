import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re, math
import random
from random import randint
from collections import Counter
from random import shuffle
import pickle
import os

UNK_idx = 0
PAD_idx = 1
EOS_idx = 2
SOS_idx = 3
use_persona = False
save_path = "data/save/"
data_path = "data/ConvAI2/"


class Lang:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {UNK_idx: "UNK", PAD_idx: "PAD", EOS_idx: "EOS", SOS_idx: "SOS"} 
        self.n_words = 4 # Count default tokens
      
    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def read_langs(file_name, cand_list, max_line = None):
    print(("Reading lines from {}".format(file_name)))
    # Read the file and split into lines
    persona = []
    dial = []
    lock = 0
    index_dial = 0
    data = {}
    with open(file_name, encoding='utf-8') as fin:
        for line in fin:
            line=line.strip()
            nid, line = line.split(' ', 1)          
            if(int(nid)==1 and lock==1):
                if(str(sorted(persona)) in data):
                    data[str(sorted(persona))].append(dial)
                else:
                    data[str(sorted(persona))] = [dial]
                persona = []
                dial = []
                lock = 0
                index_dial = 0
            lock = 1
            if '\t' in line:
                u, r, _, cand  = line.split('\t')   
                cand = cand.split('|')
                # shuffle(cand)
                for c in cand:
                    if c in cand_list:
                        pass
                    else:
                        cand_list[c] = 1 
                dial.append( {"nid":index_dial,"u":u,"r":r,'cand':cand} )
                index_dial += 1
            else:
                r = line.split(":")[1][1:-1]
                persona.append(str(r))      
    return data

def filter_data(data, cut): 
    print("Full data:",len(data))
    newdata = {}
    cnt = 0
    for k,v in data.items():
        # print("PERSONA",k)
        # print(pp.pprint(v))
        if(len(v)>cut):
            cnt+=1 
            newdata[k] = v
        # break
    print("Min {} dialog:".format(cut),cnt)
    return newdata

def cluster_persona(data, split):
    if split not in ['train', 'valid', 'test']:
        raise ValueError("Invalid split, please choose one from train, valid, test")
    filename = data_path + split +'_persona_map'
    with open(filename,'rb') as f:
        persona_map = pickle.load(f)
    # persona_map = {persona_index: [similar personas list], }
    newdata = {}
    for k, v in data.items():
        p = eval(k)
        persona_index = 0
        for p_index, p_set in persona_map.items():
            if p in p_set:
                persona_index = p_index
        if persona_index in newdata:
            for dial in v.values():
                newdata[persona_index][len(newdata[persona_index])] = dial  
        else:
            newdata[persona_index] = v
    return newdata

def preprocess(data, vocab):
    newdata = {}
    cnt_ptr = 0
    cnt_voc = 0
    for k, v in data.items():
        p = eval(k)
        
        for e in p: vocab.index_words(e)
        new_v = {i: [] for i in range(len(v))}
        for d_index, dial in enumerate(v):
            if(use_persona):
                context = list(p) 
            else:
                context = []

            for turn in dial:
                context.append(turn["u"])
                vocab.index_words(turn["u"])
                vocab.index_words(turn["r"])
                for i, c in enumerate(turn['cand']):
                    vocab.index_words(c)
                    if(turn["r"]==c): answer = i 
                        
                new_v[d_index].append([list(context), turn['cand'], answer, eval(k)])

                # print(sum(context,[]).split(" "))
                ## compute stats
                for key in turn["r"].split(" "):
                    index = [loc for loc, val in enumerate(" ".join(context).split(" ")) if (val == key)]
                    if (index):
                        cnt_ptr += 1
                    else:
                        cnt_voc += 1 
                context.append(turn["r"])
        newdata[k] = new_v
    print("Pointer percentace = {} ".format(cnt_ptr / (cnt_ptr + cnt_voc)))
    return newdata

def prepare_data_seq():
    file_train = data_path + 'train_self_original.txt'
    file_dev = data_path + 'valid_self_original.txt'
    file_test = data_path + 'test_self_original.txt'
    cand = {}
    train = read_langs(file_train, cand_list=cand, max_line=None)
    valid = read_langs(file_dev, cand_list=cand, max_line=None)
    test = read_langs(file_test, cand_list=cand, max_line=None)
    vocab = Lang()
    train = preprocess(train,vocab) #{persona:{dial1:[[context,canditate,answer,persona],[context,canditate,answer,persona]]}, dial2:[[context,canditate,answer,persona],[context,canditate,answer,persona]]}}
    valid = preprocess(valid,vocab)
    test = preprocess(test,vocab)
    train = filter_data(cluster_persona(train, 'train'),cut=1)
    valid = filter_data(cluster_persona(valid, 'valid'),cut=1)
    test = filter_data(cluster_persona(test, 'test'),cut=1)
    print("Vocab_size %s " % vocab.n_words)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(save_path + 'dataset.p', "wb") as f:
        pickle.dump([train, valid, test, vocab], f)
        print("Saved PICKLE")
    
    return train, valid, test, vocab

def laod_dataset():
    # # build from raw
    # train, valid, test, vocab = prepare_data_seq()

    with open(save_path + 'dataset.p', 'rb') as f:
        train, valid, test, vocab = pickle.load(f)

    return train, valid, test, vocab


if __name__=='__main__':
    # cluster for similar persona, each persona has several dialogs.
    # dataset = {cluster_idx: {persona_idx: [[context, candidate, answer, persona], ], }, }
    # train, valid, test, vocab = laod_dataset()
    # print(len(train))
    # print(len(valid))
    # print(len(test))

    # # each cluster has about 7.8 personas
    # avg_length = []
    # print(len(train))
    # for i in train:
    #     avg_length.append(len(train[i]))
    # print(sum(avg_length) / len(avg_length))

    # # each persona has about 7.25 dialogs.
    # avg_length = []
    # n = 3
    # for i in train[n]:
    #     for j in train[n][i]:
    #         print(j[-1])
    #     print()
    #     avg_length.append(len(train[n][i]))
    # print(sum(avg_length) / len(avg_length))


    # train, valid, test, vocab = laod_dataset()

    # for i in train:
    #     for j in train[i]:
    #         for k in range(len(train[i][j])):
    #             train[i][j][k] = [train[i][j][k][0], train[i][j][k][1][train[i][j][k][2]], train[i][j][k][3]]
    
    # for i in test:
    #     for j in test[i]:
    #         for k in range(len(test[i][j])):
    #             test[i][j][k] = [test[i][j][k][0], test[i][j][k][1][test[i][j][k][2]], test[i][j][k][3]]

    # for i in valid:
    #     for j in valid[i]:
    #         for k in range(len(valid[i][j])):
    #             valid[i][j][k] = [valid[i][j][k][0], valid[i][j][k][1][valid[i][j][k][2]], valid[i][j][k][3]]

    # with open(save_path + 'few_shot.pickle', "wb") as f:
    #     pickle.dump([train, valid, test], f)

    # with open(save_path + 'few_shot.pickle', "rb") as f:
    #     train, valid, test = pickle.load(f)

    # for i in train:
    #     for j in train[i]:
    #         print(train[i][j])
    #         exit()

    # new_train = {}
    # for i in train:
    #     for j in train[i]:
    #         if j not in new_train:
    #             new_train[j] = []
    #         for k in train[i][j]:
    #             temp = [k[0], k[1][k[2]], k[3]]
    #             new_train[j].append(temp)

    # new_test = {}
    # for i in test:
    #     for j in test[i]:
    #         if j not in new_test:
    #             new_test[j] = []
    #         for k in test[i][j]:
    #             temp = [k[0], k[1][k[2]], k[3]]
    #             new_test[j].append(temp)
    
    # new_val = {}
    # for i in valid:
    #     for j in valid[i]:
    #         if j not in new_val:
    #             new_val[j] = []
    #         for k in valid[i][j]:
    #             temp = [k[0], k[1][k[2]], k[3]]
    #             new_val[j].append(temp)

    # with open(save_path + 'zero_shot.pickle', "wb") as f:
    #     pickle.dump([new_train, new_val, new_test], f)

    with open(save_path + 'zero_shot.pickle', "rb") as f:
        train, valid, test = pickle.load(f)
    
    # print(train.keys())
    # print(train[0])

    # {person_id1: [dialog1, dialog2, ...], }
    # dialog1 = [history, response, persona]
    # for j in train[0]:
    #     print(j)
    #     print()
    #     exit()

    print(train[0][4])