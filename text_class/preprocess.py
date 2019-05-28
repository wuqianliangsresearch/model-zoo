# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 16:34:50 2019

@author: qianliang


"""
from __future__ import unicode_literals, print_function, division
from io import open

import json
import nltk
import numpy as np
import torch
import torch.utils.data as Data
import torch.nn as nn

torch.manual_seed(1)     # reproducible
MINIBATCH_SIZE = 64    # mini batch size


vcab_frq_limit = 100

def writeDicToFile(dic,filename):
    fw = open(filename,'w')
    jsObj = json.dumps(dic)
    fw.write(jsObj) 
    fw.close()
    
def readDicFromFile(filename):
    with open(filename, 'r') as f:
    	data = json.load(f)
    return data

def writeListToFile(ll, filename):

    fileObject = open(filename,'w',encoding="utf-8")
    for word in ll:
        fileObject.write(word)
        fileObject.write('\n')
    fileObject.close()
        
def readListFromFile(filename):
    
    ret = []
    with open(filename,"rb",encoding="utf-8",errors='ignore') as f:
        data = f.readlines()
        for line in data:
            word = line.strip() #list
            ret.append(word)
        
    return ret

unknown_token = "UNK"

class LangModel:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {"SOS":0, "EOS":1, "UNK":2}
        self.index2word = {0: "SOS", 1: "EOS",2: "UNK"}
        self.n_words = 3  # Count SOS , EOS and UNK
        self.x = [] # review
        self.y = [] # rating
        
    def addSentence(self, sentence, train = False):
        
        tokened_sentence = nltk.word_tokenize(sentence)
        if train:
            return tokened_sentence
        
        for word in tokened_sentence:
            self.addWord(word)
        return tokened_sentence

    def addWord(self, word):
        if word not in self.word2count:

            self.word2count[word] = 1 
            
        else:
            self.word2count[word] += 1
            # We only retain words appearing more than
            # 5 times in building the vocabulary and replace the
            # words that appear 5 times with a special UNK token
            
        if  self.word2count[word] > vcab_frq_limit and word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1
            
    
lm = LangModel("yelp")

def prepareDic(filename):
    print("Generating dic...")

    f = open(filename, encoding='utf-8')
    line = f.readline()
    while line:
        dicline = json.loads(line)
        tokened_sentence = lm.addSentence(dicline["text"])
        line = f.readline()
    f.close()

    writeDicToFile(lm.word2index,"word2index.dic")
    writeDicToFile(lm.index2word,"index2word.dic")


#prepareDic("./sample5000.json")

lm = LangModel("tain")
lm.index2word = readDicFromFile("index2word.dic")
lm.word2index = readDicFromFile("word2index.dic")

def getBatchData(filename, batch_num=64):
    print("Reading lines...")

    # Read the file and split into lines

    f = open(filename, encoding='utf-8')
    
    line = f.readline()
    x_lengths = []
    
    while line:
        
        dataline = json.loads(line)
        tokened_sentence = lm.addSentence(dataline["text"],True)
        x_lengths.append(len(tokened_sentence))
        lm.x.append(tokened_sentence)
        lm.y.append(dataline["stars"])
        
        if len(x_lengths) == batch_num:
  
            l_no = 0
            x = np.zeros((len(x_lengths), max(x_lengths)))
            for sent in lm.x:
                # replace unk and to index
                tokens = [lm.word2index[w] if w in lm.word2index else 2 for w in sent]
                for i in range(len(tokens)):
                    x[l_no, i] = int(tokens[i])
                l_no += 1

            yield x_lengths, x,lm.y
            lm.x = []
            lm.y = []
            x_lengths = []
            
        line = f.readline()
        
    f.close()
    

for x_lengths, x, y in getBatchData("./sample5000.json", MINIBATCH_SIZE):
    
    print(x_lengths, x, y)
    
    break




    

