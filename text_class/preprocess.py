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

import nltk
import nltk.data
 
def splitSentence(paragraph):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(paragraph)
    return sentences



torch.manual_seed(1)     # reproducible
#MINIBATCH_SIZE = 64    # mini batch size
unknown_token = "UNK"

vcab_frq_limit = 5

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

class LangModel:
    def __init__(self, name):
        self.name = name
        self.word2index = {"SOS":0, "EOS":1, "UNK":2}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS",2: "UNK"}
        self.n_words = 3  # Count SOS , EOS and UNK
        self.x = [] # review
        self.y = [] # rating
        
    def addSentence(self, sentence, trainOrDic = False):
        
        
        if trainOrDic:
            ret_token_sentences = []
            sents = splitSentence(sentence.replace("\n\n",""))
            for sent in sents:
                tokened_sentence = nltk.word_tokenize(sent)
                ret_token_sentences.append(tokened_sentence)
            return ret_token_sentences
        
        tokened_sentence = nltk.word_tokenize(sentence)
        
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
            
    


def prepareDic(filename,lmodel):
    print("Generating dic...")

    f = open(filename, encoding='utf-8')
    line = f.readline()
    while line:
        dicline = json.loads(line)
        tokened_sentence = lmodel.addSentence(dicline["text"])
        line = f.readline()
    f.close()

    writeDicToFile(lmodel.word2index,"word2index.dic")
    writeDicToFile(lmodel.index2word,"index2word.dic")


def main():
    lm = LangModel("yelp")
    prepareDic("./sample5000.json",lm)
  
if __name__ == '__main__':
  main()





    

