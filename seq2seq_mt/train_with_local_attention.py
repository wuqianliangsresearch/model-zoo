# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 16:34:50 2019

@author: qianliang

train data from https://nlp.stanford.edu/projects/nmt/
pytorch version of "Effective Approaches to Attention-based Neural Machine Translation"
this  file mainly focus on local attention
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
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

source_lstm_dropout = 0.2
input_reverse = True
_attn_model = "local"



SOS_token = 0
EOS_token = 1

# we filter out sentence pairs whose lengths exceed 50 words

MAX_LENGTH = 50
hidden_size = 1000
_num_layers = 4
_feed_input = True



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



class EncoderRNN(nn.Module):
    
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_dim = hidden_size

        self.embedding = nn.Embedding(input_size, self.hidden_dim)
        
        self.rnn = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers = _num_layers, 
                           dropout = source_lstm_dropout)
       
        
    def forward(self, input, hidden = None, C = None):
             
        # **input** of shape `(seq_len, batch, input_size)`
        # **h_0** of shape `(num_layers * num_directions, batch, hidden_size)`
        # **c_0** of shape `(num_layers * num_directions, batch, hidden_size)`
        
        output = self.embedding(input).view(1, 1, self.hidden_dim)
        #**h_n** of shape `(num_layers * num_directions, batch, hidden_size)`
        #**c_n** of shape `(num_layers * num_directions, batch, hidden_size)`
        output, (hidden, C_enc) = self.rnn(output, (hidden,C))
        
        # calc Context
       
        return output, (hidden, C_enc)

    def initHidden(self):
        return torch.zeros(_num_layers, 1, self.hidden_dim, device=device),torch.zeros(_num_layers, 1, self.hidden_dim, device=device)


class DecoderAttentionRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderAttentionRNN, self).__init__()
        
        self.input_dim = output_size
        self.hidden_dim = hidden_size
        self.output_dim = output_size
        
        self.D =2
        
        self.tanh = nn.Tanh()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
        self.embedding = nn.Embedding(self.input_dim, self.hidden_dim)
        if _feed_input:
            self.rnn = nn.LSTM(2*self.hidden_dim, self.hidden_dim, num_layers = _num_layers, 
                           dropout = source_lstm_dropout)
        else:
            self.rnn = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers = _num_layers, 
                           dropout = source_lstm_dropout)
        
        self.Wa = nn.Linear(self.hidden_dim, MAX_LENGTH)
        
        if _attn_model == "global":
            self.Wc = nn.Linear(MAX_LENGTH+1, 1)
        else:#local
            self.Wc = nn.Linear(2*self.D+2, 1)
            
        self.Ws = nn.Linear(self.hidden_dim, self.output_dim)
        
        self.Wp = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        self.Vp = nn.Linear(self.hidden_dim, 1)
        
        self.Wa_local = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        init.xavier_normal_(self.Wa.weight)
        init.xavier_normal_(self.Wc.weight)
        init.xavier_normal_(self.Ws.weight)

    def forward(self, input, hidden ,C , ht_hat , enc_outputs):
        
        y = self.embedding(input).view(1, 1, self.hidden_dim)
        self.sigmoid = nn.Sigmoid()
        # feed input
        if _feed_input:
#            print(y.shape,ht_hat.shape)
            y = torch.cat((y,ht_hat),dim=2)
            
        output, (hidden, C_dec) = self.rnn(y, (hidden,C))
        
        if _attn_model == "global":
            
            # localtion attention
            # at = softmax(Waht)
            # At 1xMaxlength
#            print(hidden[_num_layers-1].shape)
            At = F.softmax(self.Wa(hidden[_num_layers-1]),dim=0)
            
            # Maxlen x hidden
            Ct = torch.t(At)*enc_outputs
            # Maxlen+1 x hidden
#            print(Ct.shape,hidden[_num_layers-1].shape)
            Ct_ht = torch.cat((Ct,hidden[_num_layers-1]),dim=0)
#            print(Ct_ht.shape)
            
            # 1x hidden
            ht_hat = self.tanh(torch.t(self.Wc(torch.t(Ct_ht))))
#            print("oo",ht_hat.shape)

        else: #"local" 
            #Predictive alignment (local-p)
            
            Pt = torch.floor(MAX_LENGTH*self.sigmoid(self.Vp(F.tanh(self.Wp(hidden[_num_layers-1])))))
            Pt = torch.min(torch.max(torch.tensor([self.D*1.0],requires_grad=True).cuda(),Pt),torch.tensor([MAX_LENGTH*1.0-1-self.D*1.0],requires_grad=True).cuda()).cuda().requires_grad_()
            lb = Pt - self.D
            rb = Pt + self.D
                   
            
            lbindex = lb[0][0].int().item()
            rbindex = rb[0][0].int().item()
            enc_out = enc_outputs[lbindex:rbindex+1,]
            print("after Pt,lb:rb",Pt,lbindex,rbindex)
            #at(s) Wa_local
            # nx2D+1
            at_s = F.softmax(hidden[_num_layers-1].mm(torch.t(self.Wa_local(enc_out))))
            
            # 1x2D+1
            #s 
            s_arr = torch.arange(lbindex, rbindex+1).float().cuda()
            disFactor = torch.exp(-1.0*((s_arr -Pt)**2/(2*(self.D*1.0/2)**2))).view(1,-1)
            
            # n x 2D+1  
            Align_ht_hs = at_s*disFactor
            # 2D+1 X n
            Ct = torch.t(Align_ht_hs)*enc_out
            # 2D+2 X n 
            Ct_ht = torch.cat((Ct,hidden[_num_layers-1]),dim=0)
#            print(enc_out.shape, at_s.shape, disFactor.shape, Align_ht_hs.shape, Ct.shape, Ct_ht.shape)
            # 
            # 1x hidden
            ht_hat = self.tanh(torch.t(self.Wc(torch.t(Ct_ht))))
            
        output = self.logsoftmax(self.Ws(ht_hat))

        return output, (hidden, C_dec), ht_hat.unsqueeze(0)

    def initHidden(self):
        return torch.zeros(_num_layers, 1, self.hidden_dim, device=device),\
        torch.zeros(_num_layers, 1, self.hidden_dim, device=device),\
        torch.zeros(1, 1, self.hidden_dim, device=device)

    
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



#Cout = nn.Linear( hidden_size, C_LEN).cuda()
#init.xavier_normal(Cout.weight)
#selftanh = nn.Tanh()

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):

    enc_hidden,enc_c = encoder.initHidden() 
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    # Hj array for all Tx
    enc_outputs = torch.zeros(max_length, encoder.hidden_dim, device=device)

    loss = 0

    
    for ei in range(input_length):
        
        if input_reverse:
                
            enc_output, (enc_hidden, enc_c) = encoder(input_tensor[input_length - ei -1], enc_hidden, enc_c)
        else:
            
            enc_output, (enc_hidden, enc_c) = encoder(input_tensor[ei], enc_hidden, enc_c)
	#print(encoder_output,encoder_hidden)
#        print(enc_output.shape)
        enc_outputs[ei] = enc_hidden[_num_layers-1,0]
    
    dec_hidden,dec_c, dec_ht_hat = decoder.initHidden() 
    
    decoder_input = torch.tensor([[SOS_token]], device=device)


#    decoder_attentions = torch.zeros(max_length, max_length)
    
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    use_teacher_forcing =True
    
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, (dec_hidden,dec_c),dec_ht_hat = decoder(decoder_input, dec_hidden, dec_c, dec_ht_hat, enc_outputs)
            #decoder_attentions[di] = decoder_attention.squeeze(1)
            
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, (dec_hidden,dec_c),dec_ht_hat= decoder(decoder_input, dec_hidden, dec_c, dec_ht_hat, enc_outputs)
#            decoder_attentions[di] = decoder_attention.squeeze(1)
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
    
        encoder_outputs = torch.zeros(max_length, 2*encoder.hidden_size, device=device)
    
        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)
        
        for ei in range(input_length):
            
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]
    
    
#        C = selftanh(Cout(encoder_hidden)).cuda()
#        
#        decoder_hidden = C
        
        decoder_input = torch.tensor([[SOS_token]], device=device)
    
        for di in range(max_length):
            
            decoder_output, decoder_hidden ,decoder_attention = decoder( decoder_input, decoder_hidden, encoder_outputs )
            decoder_attentions[di] = decoder_attention.squeeze(1)
            
            topv, topi = decoder_output.data.topk(1)
            # 下一轮输入，纯值类型
            decoder_input = topi.squeeze().detach()  # detach from history as input
            
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])
                
        return decoded_words,decoder_attentions[:di+1]

def evaluateRandomly(encoder, decoder, n=50):
    for i in range(n):
        pair = random.choice(pairs)
        print('input = ', pair[0])
        print('true = ', pair[1])
        output_words,_ = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print(' predicted = ', output_sentence)
        print('')



encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder1 = DecoderAttentionRNN(hidden_size, output_lang.n_words).to(device)

trainIters(encoder1, decoder1, 100000, print_every=2000)

evaluateRandomly(encoder1, decoder1)

output_words, attentions = evaluate(encoder1, decoder1, "je suis trop froid .")
plt.matshow(attentions.numpy())











