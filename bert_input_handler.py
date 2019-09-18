#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: bert_input_handler.py
# @author: chrhad
# Input handler to tokenize and indexify text
import sys
import torch
from pytorch_pretrained_bert import BertTokenizer

PAD_TOK = "[PAD]"
MAXLEN = 512

def split_tuple_list(tupseq):
    return map(lambda i:list(map(lambda x:x[i], tupseq)), range(3))

class BertInputHandler:
    def __init__(self, pretrained_model_name):
        do_lower_case = pretrained_model_name.endswith("-uncased")
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name,
                do_lower_case=do_lower_case)
   
    """
    Tokenize left, head, and right and convert to vocab index in the BERT vocab
    """
    def tokenize_indexify(self, triplet, bos=None, maxlen=MAXLEN):  # beginning of sentence, e.g. [CLS]
        if maxlen <= 0:
            maxlen = MAXLEN
        # sentence is a list of list of tuples
        left, sentence, right = triplet
        left_tups = list(self.tokenize_tuples(left))
        sentence_tups = list(self.tokenize_tuples(sentence))
        right_tups = list(self.tokenize_tuples(right))
        while len(left_tups) + len(sentence_tups) + len(right_tups) >= maxlen and len(left_tups) + len(right_tups) > 0:
            if len(left_tups) > 0:
                left_tups.pop(0)
            if len(right_tups) > 0:
                right_tups.pop(-1)
        if bos is not None and len(bos) > 0:
            left_tups = [(bos, '#', '#')] + left_tups
        tokens, all_lexels, all_iids = tuple(split_tuple_list(left_tups + sentence_tups + right_tups))
        tokens_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens),
                dtype=torch.long)
        lexels = [lex for lex in all_lexels if (lex != '*' and lex != '#')]
        iids = [iid for iid in all_iids if (iid != '*' and iid != '#')]
        head_offsets = torch.tensor([i for (i, x) in enumerate(all_iids) if (x != '*' and x!= '#')], dtype=torch.long)
        return (tokens_ids, lexels, iids, head_offsets, all_lexels)
    
    def tokenize_tuples(self, sentence):
        for tup in sentence:
            subtoks = self.tokenizer.tokenize(tup[0])
            yield (subtoks[0], tup[1], tup[2])
            for t in subtoks[1:]:
                yield (t, '*', '*')
    
    def pad_idx(self):
        return self.tokenizer.convert_tokens_to_ids([PAD_TOK])[0]
