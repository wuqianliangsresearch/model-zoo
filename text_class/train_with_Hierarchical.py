# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 16:34:50 2019

@author: qianliang

pytorch version of "Hierarchical Attention Networks for Document Classification"
"""
from __future__ import unicode_literals, print_function, division
from io import open
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import random

import torch
import torch.nn as nn
from torch.nn import init
from torch import optim
from preprocess import *
torch.manual_seed(1) 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH = 30
MINIBATCH_SIZE = 64 
HIDDEN_SIZE = 50
embedding_size = 200


lm = LangModel("tain")
lm.index2word = readDicFromFile("index2word.dic")
lm.word2index = readDicFromFile("word2index.dic")


def getBatchData(filename, batch_num=64):
    print("Reading lines...")

    # Read the file and split into lines

    f = open(filename, encoding='utf-8')
    
    line = f.readline()

    while line:
        
        dataline = json.loads(line)
        tokened_sentence = lm.addSentence(dataline["text"],True)

        lm.x.append(tokened_sentence)
        lm.y.append(dataline["stars"])
        
        if len(lm.x) == batch_num:
            
            x = []
            
            for i in range(0,len(lm.x)):
                document = lm.x[i]
                xx = []
                for sentence in document:
                    xx.append([lm.word2index[w] if w in lm.word2index else 2 for w in sentence])
                x.append(xx)    
                    
            yield x,lm.y
            
            lm.x = []
            lm.y = []

        line = f.readline()
        
    f.close()
    


# Hierarchical Attention Networks
    
class HAN_word_enconder(nn.Module):
    
    def __init__(self, input_size, hidden_size = HIDDEN_SIZE):
        super(HAN_word_enconder, self).__init__()
        
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, bidirectional=True)
        self.Ww = nn.Linear(2*self.hidden_size, self.hidden_size)
        self.Uw = nn.Linear(self.hidden_size, 1)
        
        init.xavier_normal_(self.Ww.weight)
        init.xavier_normal_(self.Uw.weight)
        
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, input, hidden):

        embedded = self.embedding(input)
        output, hidden = self.gru(embedded, hidden)
        Uit = self.tanh(self.Ww(output))
        U = self.Uw(Uit).squeeze(2)
#        print(U.shape)
        A = self.softmax(U)
#        print(A.shape)
        Ait = A.view(-1,1,1)
#        print(Ait.shape)
        Si = torch.sum(Ait*output,0)

        return Si, hidden

    def initHidden(self):
        return torch.zeros(2, 1, self.hidden_size, device=device)


class HAN_sentence_enconder(nn.Module):
    
    def __init__(self, input_size, hidden_size = HIDDEN_SIZE):
        super(HAN_sentence_enconder, self).__init__()
        
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, bidirectional=True)
        
        self.Ws = nn.Linear(2*self.hidden_size, self.hidden_size)
        self.Us = nn.Linear(self.hidden_size, 1)
        self.Wc = nn.Linear(2*self.hidden_size, 5)
        
        init.xavier_normal_(self.Ws.weight)
        init.xavier_normal_(self.Us.weight)
        
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=0)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
    def forward(self, input, hidden):
        
        output, hidden = self.gru(input, hidden)
        Uit = self.tanh(self.Ws(output))
        U = self.Us(Uit).squeeze(2)
        Ait = self.softmax(U).view(-1,1,1)
        Vi = torch.sum(Ait*output,0)
        
        output = self.logsoftmax(self.Wc(Vi)) 
        return output, hidden

    def initHidden(self):
        return torch.zeros(2, 1, self.hidden_size, device=device)


encoder1 = HAN_word_enconder(len(lm.word2index)+1, HIDDEN_SIZE).cuda()
encoder2 = HAN_sentence_enconder(2*HIDDEN_SIZE, HIDDEN_SIZE).cuda()

def tensorFromSentence( sentenceIndexes):

    return torch.tensor(sentenceIndexes, dtype=torch.long, device=device).view(-1, 1).cuda()

def train(x, y, encoder1, encoder2, encoder1_optimizer, encoder2_optimizer, criterion, max_length=MAX_LENGTH):

    encoder1_hidden = encoder1.initHidden() 
    encoder2_hidden = encoder2.initHidden() 
    
    encoder1_optimizer.zero_grad()
    encoder2_optimizer.zero_grad()

    
    loss = 0
    
    for i in range(0,len(x)):
        doc = x[i]
        label = y[i]
        label_tensor = torch.zeros(1,dtype=torch.long).cuda()
        label_tensor[0] = math.floor(label)-1
        if label_tensor[0] < 0:
            label_tensor[0] = 0  #下标和分数相差1
        
        
        word_enc_outputs = torch.zeros(len(doc), 1, 2*encoder1.hidden_size, device=device)
        for j in range(0,len(doc)):
            sent_tensor = tensorFromSentence(doc[j])
            
            # whole sentence input
            Si, encoder1_hidden_t_at_seql = encoder1(sent_tensor, encoder1_hidden)
            word_enc_outputs[j] = Si
        # whole doc si input
        encoder2_output, encoder2_hidden = encoder2(word_enc_outputs, encoder2_hidden)
        #print(encoder2_output,label_tensor)
        loss += criterion(encoder2_output, label_tensor)
        
    loss.backward()
    
    # Clip gradients: gradients are modified in place
    _ = torch.nn.utils.clip_grad_norm_(encoder1.parameters(), 10)
    _ = torch.nn.utils.clip_grad_norm_(encoder2.parameters(), 10)
    
    encoder1_optimizer.step()
    encoder2_optimizer.step()

    return loss.item()/len(y) 


import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def trainIters(encoder1, encoder2, n_iters=1000, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

#    encoder_optimizer = optim.RMSprop(encoder.parameters(), lr=learning_rate, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
#    decoder_optimizer = optim.RMSprop(decoder.parameters(), lr=learning_rate, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    lrr_de = 1.0
    encoder1_optimizer = optim.SGD(encoder1.parameters(), lr=learning_rate*lrr_de)
    encoder2_optimizer = optim.SGD(encoder2.parameters(), lr=learning_rate*lrr_de)
#    print("learning rate:",learning_rate*lrr_de) 

    criterion = nn.NLLLoss()

    iter =1
    for  x, y in getBatchData("./yelp_academic_dataset_review.json",32):
        

        loss = train(x, y, encoder1, encoder2, encoder1_optimizer, encoder2_optimizer, criterion)
        #print(iter, loss)
        
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
#       	lrr_de = iter / n_iters *1.0
        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            
        iter +=1



trainIters(encoder1, encoder2, n_iters = 1000 , print_every=50)










