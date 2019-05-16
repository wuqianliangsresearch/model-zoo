# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 16:34:50 2019

@author: qianliang

pytorch version of "NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE"
"""
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata

import re
import random

import torch
import torch.nn as nn
from torch.nn import init
from torch import optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    # 标点符号，用空格隔开。
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('./%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs
    
MAX_LENGTH = 100

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
aa  = random.choice(pairs)
print(aa)

C_LEN = 1024

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.Tanh = nn.Tanh()
        
    def forward(self, input, hidden):
        
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
       
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderAttentionRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderAttentionRNN, self).__init__()
        
        self.input_dim = output_size
        self.hidden_dim = hidden_size
        self.output_dim = output_size
        self.const_C_from_encoder = None

        self.softmax = nn.LogSoftmax(dim=1)
        self.a_sm = nn.LogSoftmax(dim=0)
        
        self.Uz = nn.Linear( self.input_dim, self.hidden_dim)
        self.Wz = nn.Linear( self.hidden_dim, self.hidden_dim)
        self.Cz = nn.Linear( self.hidden_dim, self.hidden_dim)
        
        self.Ur = nn.Linear( self.input_dim, self.hidden_dim)
        self.Wr = nn.Linear( self.hidden_dim, self.hidden_dim)
        self.Cr = nn.Linear( self.hidden_dim, self.hidden_dim)
        
        self.Uh = nn.Linear( self.input_dim, self.hidden_dim)
        self.Wh = nn.Linear( self.hidden_dim, self.hidden_dim)
        self.Ch = nn.Linear( self.hidden_dim, self.hidden_dim)
        
        self.Ua = nn.Linear( self.hidden_dim, self.hidden_dim)
        self.Wa = nn.Linear( self.hidden_dim, self.hidden_dim)
        self.Va = nn.Linear( self.hidden_dim, self.hidden_dim)
        
        
        
        
        
        self.V = nn.Linear( self.hidden_dim, self.output_dim)


        self.Cin = nn.Linear(C_LEN, self.hidden_dim)

        self.Tanh = nn.Tanh()
        self.Sigmoid = nn.Sigmoid()
        
        init.xavier_normal_(self.Wz.weight)
        init.xavier_normal_(self.Uz.weight)
        init.xavier_normal_(self.Cz.weight)
        
        init.xavier_normal_(self.Wr.weight)
        init.xavier_normal_(self.Ur.weight)
        init.xavier_normal_(self.Cr.weight)
        
        init.xavier_normal_(self.Wh.weight)
        init.xavier_normal_(self.Uh.weight)
        init.xavier_normal_(self.Ch.weight)
        
        init.xavier_normal_(self.Wa.weight)
        init.xavier_normal_(self.Ua.weight)
        init.xavier_normal_(self.Va.weight)
        
        
        
        init.xavier_normal_(self.V.weight)
        
        init.xavier_normal_(self.Cin.weight)
        
        self.embedding = nn.Embedding(self.output_dim, self.output_dim)
        
        

    def forward(self, input, hidden , encoder_outputs):
        
        x = self.embedding(input).view(1, 1, -1).cuda()
        
        if hidden is None:
            hidden = self.Wh(encoder_outputs[0]).view(1,1,-1).cuda()
        
        Anxn = self.Va(self.Tanh(self.Wa(hidden)+self.Ua(encoder_outputs)))
        weighted_a = self.a_sm(Anxn)*encoder_outputs

        Ci =torch.sum(weighted_a[0],dim=0).view(1,1,self.hidden_dim)
        
        # GRU Layer 1
        z_t1 = self.Sigmoid(self.Uz(x) + self.Wz(hidden) + self.Cz(Ci))
        r_t1 = self.Sigmoid(self.Ur(x) + self.Wr(hidden) + self.Cr(Ci))
        # h ~
        c_t1 = self.Tanh(self.Uh(x) + self.Wh(r_t1*hidden) + self.Ch(Ci) )
        hidden = (torch.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * hidden
        
        output = self.V(hidden)
        output = self.softmax(output[0])
        
        return output, hidden


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


teacher_forcing_ratio = 0.5

hidden_size = 256

#Cout = nn.Linear( hidden_size, C_LEN).cuda()
#init.xavier_normal(Cout.weight)
#selftanh = nn.Tanh()

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):

    encoder_hidden = encoder.initHidden() 
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    # Hj array for all Tx
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

#    print('encoder_outputs.shape',encoder_outputs.shape)
#    C = selftanh(Cout(encoder_hidden)).cuda()
#    
#    decoder_hidden = C
    decoder_hidden = None
    
    decoder_input = torch.tensor([[SOS_token]], device=device)



    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    use_teacher_forcing =True
    
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder( decoder_input, decoder_hidden,encoder_outputs )
            
            topv, topi = decoder_output.topk(1)
            # 下一轮输入，纯值类型
            decoder_input = topi.squeeze().detach()  # detach from history as input
            
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()
    
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


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


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

#    encoder_optimizer = optim.RMSprop(encoder.parameters(), lr=learning_rate, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
#    decoder_optimizer = optim.RMSprop(decoder.parameters(), lr=learning_rate, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    lrr_de = 1.0
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate*lrr_de)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate*lrr_de)
    print("learning rate:",learning_rate*lrr_de) 
    training_pairs = [tensorsFromPair(random.choice(pairs)) for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        #print(iter, loss)
        
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
       	lrr_de = iter / n_iters *1.0
        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)
    
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):

    with torch.no_grad():
        
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        
        encoder_hidden = encoder.initHidden()
        decoder_hidden = None
    
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    
        decoded_words = []
        
        for ei in range(input_length):
            
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]
    
    
#        C = selftanh(Cout(encoder_hidden)).cuda()
#        
#        decoder_hidden = C
        
        decoder_input = torch.tensor([[SOS_token]], device=device)
    
        for di in range(max_length):
            
            decoder_output, decoder_hidden = decoder( decoder_input, decoder_hidden, encoder_outputs )
                
            topv, topi = decoder_output.data.topk(1)
            # 下一轮输入，纯值类型
            decoder_input = topi.squeeze().detach()  # detach from history as input
            
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])
                
        return decoded_words

def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('input = ', pair[0])
        print('true = ', pair[1])
        output_words = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print(' predicted = ', output_sentence)
        print('')



encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder1 = DecoderAttentionRNN(hidden_size, output_lang.n_words).to(device)

trainIters(encoder1, decoder1, 100, print_every=50)

evaluateRandomly(encoder1, decoder1)

