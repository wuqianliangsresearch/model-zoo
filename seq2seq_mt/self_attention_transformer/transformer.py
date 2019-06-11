# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 11:18:38 2019

@author: qianliang
"""

from __future__ import unicode_literals, print_function, division

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import Constants as Constants

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super( ScaledDotProductAttention, self ).__init__() 
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """前向传播.

        Args:
            q: Queries张量，形状为[B, L_q, D_q]
            k: Keys张量，形状为[B, L_k, D_k]
            v: Values张量，形状为[B, L_v, D_v]，一般来说就是k
            scale: 缩放因子，一个浮点标量
            attn_mask: Masking张量，形状为[B, L_q, L_k]

        Returns:
            上下文张量和attetention张量
        """
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask is not None:
            # 给需要mask的地方设置一个负无穷
            attention = attention.masked_fill_(attn_mask, -np.inf)
        # 计算softmax
        attention = self.softmax(attention)
        # 添加dropout
        attention = self.dropout(attention)
        # 和V做点积
        context = torch.bmm(attention, v)
        return context, attention

class MultiHeadAttentionNew(nn.Module):
    ''' Multi-Head Attention module '''
    
    def __init__(self, d_model=512, n_head=8, dropout=0.0):
    #def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention,self).__init__()
        
        self.n_head = n_head
        self.d_k = int(d_model/n_head)
        self.d_v = int(d_model/n_head)

        self.w_qs = nn.Linear(d_model, d_model) 
        self.w_ks = nn.Linear(d_model, d_model)
        self.w_vs = nn.Linear(d_model, d_model)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + self.d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + self.d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + self.d_v)))

        self.attention = ScaledDotProductAttention(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(d_model, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()
        print(len_q,len_k,len_v)

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, np.power(self.d_k, 0.5), mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn
    
class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):

        super(MultiHeadAttention,self).__init__()
        self.dim_per_head = model_dim // num_heads  # 64
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
		# multi-head attention之后需要做layer norm
        self.layer_norm = nn.LayerNorm(model_dim)

        nn.init.normal_(self.linear_k.weight, mean=0, std=np.sqrt(2.0 / (model_dim + self.dim_per_head)))
        nn.init.normal_(self.linear_v.weight, mean=0, std=np.sqrt(2.0 / (model_dim + self.dim_per_head)))
        nn.init.normal_(self.linear_q.weight, mean=0, std=np.sqrt(2.0 / (model_dim + self.dim_per_head)))

        nn.init.xavier_normal_(self.linear_final.weight)
        
    def forward(self, key, value, query, attn_mask=None):
		# 残差连接
        residual = query

        num_heads = self.num_heads
        dim_per_head = self.dim_per_head
        batch_size, len_q, _ = query.size()
        batch_size, len_k, _ = key.size()
        batch_size, len_v, _ = value.size()
        
        scale = (key.size(-1) // num_heads) ** -0.5

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        q = self.linear_q(query).view(batch_size, len_q, num_heads, dim_per_head)
        k = self.linear_k(key).view(batch_size, len_k, num_heads, dim_per_head)
        v = self.linear_v(value).view(batch_size, len_v, num_heads, dim_per_head)

        query = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, dim_per_head) # (n*b) x lq x dk
        key = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, dim_per_head) # (n*b) x lk x dk
        value = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, dim_per_head) # (n*b) x lv x dv
        
        if attn_mask is not None:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)
            
        # scaled dot product attention
        context, attention = self.dot_product_attention(
          query, key, value, scale, attn_mask)

        # concat heads
        #context = context.view(batch_size, -1, dim_per_head * num_heads)
        context = context.view(num_heads, batch_size, len_q, dim_per_head)
        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, len_q, -1) # b x lq x (n*dv)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, attention

def padding_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
	# seq_k和seq_q的形状都是[B,L]
#    print(seq_k.shape,seq_q.shape )
    len_q = seq_q.size(1)
    # `PAD` is 0
    pad_mask = seq_k.eq(Constants.PAD)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
    return pad_mask.cuda()

def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1).cuda()


def sequence_mask(seq):
    batch_size, seq_len = seq.size()
    
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8),diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, L]
    return mask.cuda()

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

class PositionalWiseFeedForward(nn.Module):

    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        super( PositionalWiseFeedForward, self ).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
#        print('x.shape:',x.shape) 
        output = x.transpose(1, 2)
        output = self.w2(F.relu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))

        # add residual and norm layer
        output = self.layer_norm(x + output)
        return output


class EncoderLayer(nn.Module):
    """Encoder的第一层。"""
    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2018, dropout=0.0):
        

        super( EncoderLayer, self ).__init__()
        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs, non_pad_mask=None, attn_mask=None):

        # self attention
        context, attention = self.attention(inputs, inputs, inputs, attn_mask)
        context *= non_pad_mask
        # feed forward network
        output = self.feed_forward(context)
        output *= non_pad_mask
        
        return output, attention


class Encoder(nn.Module):
    """多层EncoderLayer组成Encoder。"""

    def __init__(self,
               vocab_size,
               max_seq_len,
               num_layers=6,
               model_dim=512,
               num_heads=8,
               ffn_dim=2048,
               dropout=0.0):

        super( Encoder, self ).__init__()
        self.encoder_layers = nn.ModuleList(
          [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
           range(num_layers)])

        self.seq_embedding = nn.Embedding(vocab_size, model_dim, padding_idx=0)
        #self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)
        self.pos_embedding = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(max_seq_len+1, model_dim, padding_idx=0),
            freeze=True)

    def forward(self, inputs, inputs_len):

        non_pad_mask = get_non_pad_mask(inputs)
        self_attention_mask = padding_mask(inputs, inputs)
        
        output = self.seq_embedding(inputs) + self.pos_embedding(inputs_len) # 映射到 seq_len x d_model
                    

        attentions = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output, 
                non_pad_mask, 
                self_attention_mask)
            attentions.append(attention)

        return output, attentions


class DecoderLayer(nn.Module):

    def __init__(self, model_dim, num_heads=8, ffn_dim=2048, dropout=0.0):

        super( DecoderLayer, self ).__init__()
        # masked multihead
        self.slf_attn = MultiHeadAttention(model_dim, num_heads, dropout)
        self.enc_attn = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self,
              dec_inputs,
              enc_outputs,
              non_pad_mask = None,
              self_attn_mask=None,
              context_attn_mask=None):
        # self attention, all inputs are decoder inputs
        dec_output, self_attention = self.slf_attn(
          dec_inputs, dec_inputs, dec_inputs, self_attn_mask)
        dec_output *= non_pad_mask
        # context attention
        # query is decoder's outputs, key and value are encoder's inputs
        dec_output, context_attention = self.enc_attn(
          enc_outputs, enc_outputs, dec_output, context_attn_mask)
        dec_output *= non_pad_mask
        
        # decoder's output, or context
        dec_output = self.feed_forward(dec_output)
        dec_output *= non_pad_mask
        return dec_output, self_attention, context_attention


class Decoder(nn.Module):

    def __init__(self,
               vocab_size,
               max_seq_len,
               num_layers=6,
               model_dim=512,
               num_heads=8,
               ffn_dim=2048,
               dropout=0.0):

        super( Decoder, self ).__init__()
        self.num_layers = num_layers

        self.decoder_layers = nn.ModuleList(
          [DecoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
           range(num_layers)])

        self.seq_embedding = nn.Embedding(vocab_size, model_dim, padding_idx=0)
        #self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)
        self.pos_embedding = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(max_seq_len+1, model_dim, padding_idx=0),
            freeze=True)

    def forward(self, inputs, inputs_len, enc_output, context_attn_mask=None):

        non_pad_mask = get_non_pad_mask(inputs)
        self_attention_padding_mask = padding_mask(inputs, inputs)
        self_subsequent_mask = sequence_mask(inputs)
        self_attn_mask = torch.gt((self_attention_padding_mask + self_subsequent_mask), 0)
        
        output = self.seq_embedding(inputs)
        output += self.pos_embedding(inputs_len)

        self_attentions = []
        context_attentions = []
        for decoder in self.decoder_layers:
            output, self_attn, context_attn = decoder(
            output, enc_output, non_pad_mask, self_attn_mask, context_attn_mask)
            self_attentions.append(self_attn)
            context_attentions.append(context_attn)

        return output, self_attentions, context_attentions
