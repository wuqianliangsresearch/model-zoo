''' Define the Transformer model '''

import torch.nn as nn
import numpy as np
from transformer import *

__author__ = "wuqianliang"

class Transformer(nn.Module):
    ''' A sequence to sequence model with self attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, 
            n_tgt_vocab, 
            len_max_seq,
            d_word_vec=512, 
            d_model=512, 
            d_inner=2048,
            n_layers=6, 
            n_head=8,  
            dropout=0.1,
            tgt_emb_prj_weight_sharing=True,
            emb_src_tgt_weight_sharing=True):
        
        super().__init__()

        self.encoder = Encoder(
            self,
            vocab_size = n_src_vocab,
            max_seq_len = len_max_seq,
            num_layers = n_layers, 
            model_dim = d_model, 
            ffn_dim = d_inner,
            num_heads = n_head, 
            dropout = dropout)
            
        self.decoder = Decoder(
            self,
            vocab_size = n_src_vocab, 
            max_seq_len = len_max_seq,
            num_layers = n_layers, 
            model_dim = d_model, 
            ffn_dim = d_inner,
            num_heads = n_head, 
            dropout = dropout)

        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.tgt_word_prj.weight = self.decoder.seq_embedding.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

        if emb_src_tgt_weight_sharing:
            # Share the weight matrix between source & target word embeddings
            assert n_src_vocab == n_tgt_vocab, \
            "To share word embedding table, the vocabulary size of src/tgt shall be the same."
            self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight

    def forward(self, src_seq, src_lens, tgt_seq, tgt_lens):
 
        enc_output, *_ = self.encoder(src_seq, src_lens)
        
        dec_enc_attn_padding_mask = padding_mask(seq_k=src_seq, seq_q=tgt_seq)
        
        dec_output, *_ = self.decoder(tgt_seq, tgt_lens, enc_output, dec_enc_attn_padding_mask)
        seq_logit = self.tgt_word_prj(dec_output) * self.x_logit_scale

        return seq_logit.view(-1, seq_logit.size(2))
