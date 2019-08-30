# /usr/bin/env python
# -*- coding: utf-8 -*-
# File: model.py
# @author: chrhad
# BERT classifier model for various lexelts
import copy
import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
from pytorch_pretrained_bert import BertConfig, BertModel

def pad(seqs, pad_ix=0):
    maxlen = max([len(seq) for seq in seqs])
    ret = torch.zeros(len(seqs), maxlen, dtype=torch.long)
    if pad_ix != 0:
        ret.fill_(pad_ix)
    for i, seq in enumerate(seqs):
        ret[i][:len(seq)] = seq
    return ret

# Multi-headed self-attention
class MultiHeadedAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, transform_value=True, dropout=0.):
        super(MultiHeadedAttention, self).__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                    "The hidden size ({0:d}) is not a multiple of the number of attention \
                            heads ({1:d})".format(hidden_dim, num_heads))
        self.num_heads = num_heads
        self.attn_head_dim = hidden_dim // num_heads
        self.all_head_dim = self.num_heads * self.attn_head_dim

        self.query = nn.Linear(hidden_dim, self.all_head_dim)
        self.key = nn.Linear(hidden_dim, self.all_head_dim)
        self.value = nn.Linear(hidden_dim, self.all_head_dim) if transform_value \
                else None
        
        self.dropout = nn.Dropout(dropout)
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attn_head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query_in, key_in, target, attention_mask):
        mx_query_layer = self.query(query_in)
        mx_key_layer = self.key(key_in)
        mx_value_layer = self.value(target) if self.value is not None else target

        query_layer = self.transpose_for_scores(mx_query_layer)
        key_layer = self.transpose_for_scores(mx_key_layer)
        value_layer = self.transpose_for_scores(mx_value_layer)

        # Dot product between query and key
        attn_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attn_scores = attn_scores / math.sqrt(self.attn_head_dim)
        ext_attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attn_scores = attn_scores + (1. - ext_attn_mask) * -10000.
        attn_probs = F.softmax(attn_scores, dim=-1) 

        attn_probs = self.dropout(attn_probs)

        context_layer = torch.matmul(attn_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_dim = context_layer.size()[:-2] + (self.all_head_dim, )
        context_layer = context_layer.view(*new_context_layer_dim)
        return context_layer

class AddNorm(nn.Module):  # add input and output
    def __init__(self, in_dim, hidden_dim, dropout=0.):
        super(AddNorm, self).__init__()
        self.dense = nn.Linear(in_dim, hidden_dim)  # W^O (Vaswani et al, 2017)
        self.layer_norm = LayerNorm(hidden_dim, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, in_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + in_tensor)
        return hidden_states

def gelu(x):
    """Implementation of the Gaussian linear unit activation function.
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

ACT2FN = {'gelu': gelu, "relu": F.relu}

class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim, hidden_act='gelu'):
        super(PositionwiseFeedForward, self).__init__()
        self.dense = nn.Linear(hidden_dim, intermediate_dim)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        return hidden_states

class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, intermediate_dim, hidden_act='gelu',
            dropout=0.):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadedAttention(hidden_dim, num_heads, dropout)
        self.attn_output = AddNorm(hidden_dim, hidden_dim, dropout)
        self.intermediate = PositionwiseFeedForward(hidden_dim, intermediate_dim,
                hidden_act)
        self.output = AddNorm(intermediate_dim, hidden_dim, dropout)

    def forward(self, in_tensor, attention_mask):
        attn_out = self.attn(in_tensor, attention_mask)
        attn_out = self.attn_output(attn_out, in_tensor)
        inter_out = self.intermediate(attn_out)
        output = self.output(inter_out, attn_out)
        return output

# Multi-layer perceptron (MLP) with dropout
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=None, dropout=0.):
        super(MLP, self).__init__()
        affines = []
        prev = input_dim
        if hidden_dims is not None: # create hidden layers of affine transformation
            for dim in hidden_dims:
                affines.append(nn.Linear(prev, dim))
                nn.init.xavier_normal_(affines[-1].weight)
                prev = dim
        affines.append(nn.Linear(prev, output_dim))  # the last, might be the only one
        self.num_layers = len(affines)
        self.affine_seq = nn.ModuleList(affines)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for i, tr in enumerate(self.affine_seq):
            x = tr(x)
            if i < self.num_layers - 1:
                x = F.tanh(x)
                x = self.dropout(x)  # dropout only applied to intermediate layers
        return x


class BertSenseClassifier(nn.Module):
    def __init__(self, model_name, lexelt_sense_num, sense_lex_filter, mlp_dropout=0.,
            attn_dropout=0., pad_ix=0, unk_ix=0, layer=-1, use_glu=False, residual_glu=False,
            act_fn='gelu', top_attn_head=1, sent_attn_query=False, freeze_bert=False):
        super(BertSenseClassifier, self).__init__()

        # BERT parameters
        self.bert_model = BertModel.from_pretrained(model_name)
        self.hidden_size = self.bert_model.config.hidden_size
        self.maxlen = self.bert_model.config.max_position_embeddings
        for p in self.bert_model.parameters():
            p.requires_grad = False

        self.pad_ix = pad_ix
        self.unk_ix = unk_ix
        # layer-wise attention to weight different layer outputs
        self.layer = layer
        self.layer_attn = MultiHeadedAttention(self.hidden_size, top_attn_head,
                transform_value=False, dropout=attn_dropout) if layer < 0 else None
        if self.layer_attn is not None:
            self.layer_attn.apply(self.init_weights)
        self.uquery = nn.Parameter(torch.empty(1, 1, self.hidden_size, dtype=torch.float)) \
                if (layer < 0 and not sent_attn_query) else None
        if self.uquery is not None:
            self.uquery.data.normal_(mean=0.0, std=self.bert_model.config.initializer_range)

        # Sense classifier
        self.use_glu = use_glu
        self.residual_glu = residual_glu
        self.glu_gate = nn.Linear(self.hidden_size, 2 * self.hidden_size) if use_glu else None
        self.dropout = nn.Dropout(mlp_dropout)
        self.mlp_in_size = self.hidden_size
        self.mlp = MLP(self.mlp_in_size, lexelt_sense_num, dropout=mlp_dropout)

        # sense_lex_filter
        self.sense_lex_filter = nn.Parameter(torch.tensor(
            [sense_lex_filter], dtype=torch.long), requires_grad=False)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.bert_model.config.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def forward(self, sentences, offsets, lexelts, is_log=True):
        encoded = self._bert_encode(sentences)  # encoded: batch_size x maxlen x hidden_size
        #attention_mask = sentences.ne(self.pad_ix)
        maxlen = encoded.size(1)
        offset_cat = self._flatten_offsets(offsets, maxlen)
        lexelt_cat = torch.cat(lexelts)
        encoded_btflat = encoded.view(-1, encoded.size(2), self.hidden_size) # batch_size * maxlen x num_layers x hidden_size
        slices = encoded_btflat.index_select(0, offset_cat)

        if self.layer_attn is not None: # compute layerwise attention
            sent_offset_cat = offset_cat // maxlen * maxlen  # sent_offset_cat: batch_size x maxlen x num_layers x hidden_size
            query_exp = self.uquery.expand(slices.size(0), -1, -1) if self.uquery is not None \
                    else encoded_btflat.index_select(0, sent_offset_cat)[:,-1:,:]
            layer_attn_mask = torch.ones(slices.size(0), slices.size(1))
            layer_attn_mask = layer_attn_mask.to(dtype=query_exp.dtype, device=query_exp.device)
            slices = self.layer_attn(query_exp, slices, slices, layer_attn_mask).squeeze(1)
        else:
            slices = slices[:,self.layer,:]

        if self.use_glu:
            glu_in = self.dropout(slices)
            glu_in = self.glu_gate(glu_in)
            if self.residual_glu:
                glu_in[:,:slices.size(1)] = (glu_in[:,:slices.size(1)] + slices) * math.sqrt(0.5)
            glu_out = F.glu(glu_in, dim=-1)
            slices = glu_out

        logits = self.mlp(slices)
        logits = logits + F.relu(1. - self._create_mask(lexelt_cat)) * -10000.
        if is_log:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    def train_parameters(self):
        return filter(lambda p: p.requires_grad, self.parameters())
    
    def _bert_encode(self, sentences):
        # Pass to BERT model
        with torch.no_grad():
            encoded, _ = self.bert_model(sentences,
                    attention_mask=sentences.ne(self.pad_ix))
            return torch.stack(encoded, dim=2)
    
    def _create_mask(self, lexelts):
        return self.sense_lex_filter.eq(lexelts.unsqueeze(1)).float() + \
                self.sense_lex_filter.eq(self.unk_ix).float()

    def _flatten_offsets(self, offsets, maxlen):
        return torch.cat([i * maxlen + x for i, x in enumerate(offsets)])

