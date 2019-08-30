#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: test.py
# @author: chrhad
# Run BERT classifier for all word WSD task
import argparse
import io
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from bert_input_handler import BertInputHandler
from instances_reader import InputLoaderFactory
from utils import open_file
from model import BertSenseClassifier, pad


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
            "Run word sense disambiguation model for all lexical items")
    argparser.add_argument('test_path', help="Test instance file in XML format")
    argparser.add_argument('model_dir', help="Directory to pre-trained model")
    argparser.add_argument('output_dir', help="Directory to output predictions")

    # Optional arguments
    argparser.add_argument('--batch-size', type=int, default=32,
            help="Training batch size (default: 32)")
    argparser.add_argument('--xml-format', choices=[None, 'semeval13', 'senseval2ls', 'senseval3ls', 'semeval13induction'],
            default=None, help="Input XML file format (default: training model default)")
    args = argparser.parse_args()

    mdir = args.model_dir  # path to store trained model
    odir = args.output_dir  # path to store results
    if not os.path.exists(odir):
        os.mkdir(odir)
    elif not os.path.isdir(odir):
        sys.exit("{0} exists but not a directory".format(odir))

    # 0. Load model and instances, assign indices
    lexelt2ix = torch.load("{0}/lexelt_idx.bin".format(mdir))
    ix2key = torch.load("{0}/sense_idx.bin".format(mdir))
    key2lexelt = torch.load("{0}/sense_lexelt".format(mdir))
    params = torch.load("{0}/params.bin".format(mdir))

    print("| Loding BERT tokenizer from {0}".format(params.bert_model),
            file=sys.stderr, flush=True)
    input_handler = BertInputHandler(params.bert_model)
    pad_ix = input_handler.pad_idx()

    print("| Loading model from {0}".format(mdir))
    use_glu = params.use_glu if 'use_glu' in params else False
    residual_glu = params.residual_glu if 'residual_glu' in params else False
    sent_attn_query = params.sent_attn_query if 'sent_attn_query' in params else False
    model_fname = "{0}/model.bin".format(mdir)
    model = BertSenseClassifier(params.bert_model, len(ix2key), key2lexelt,
            mlp_dropout=params.dropout, attn_dropout=params.attn_dropout,
            layer=params.layer, use_glu=use_glu, residual_glu=residual_glu,
            top_attn_head=params.top_attn_head, sent_attn_query=sent_attn_query)
    model.load_state_dict(torch.load(model_fname))
    model.cuda()
    model.eval()

    xml_format = params.xml_format if args.xml_format is None else args.xml_format
    print("| Loading test instances from {0} with {1} XML format".format(args.test_path, xml_format),
            file=sys.stderr, flush=True)
    input_loader = InputLoaderFactory(xml_format)
    sentences, _ = input_loader.load(args.test_path, params.num_context)
    instances = []
    iids = []
    lexelts = []
    heads = []

    def predict_batch(instances, iids, lexelts, heads):
        batch_sents = pad(instances).cuda()
        batch_heads = [x.cuda() for x in heads]
        batch_lexelts = [x.cuda() for x in lexelts]

        lprobs = model(batch_sents, batch_heads, batch_lexelts)
        _, argmaxs = lprobs.max(dim=-1)
        i = 0  # iterate returned element
        for sent_iid in iids:
            for iid in sent_iid:
                yield (iid, ix2key[argmaxs[i]])
                i += 1

    bsz = args.batch_size
    cnt = 0
    with open_file("{0}/result.result".format(odir), 'w') as f:
        for inst in sentences:
            if cnt == bsz:
                # predict batch
                for iid, out in predict_batch(instances, iids, lexelts, heads):
                    print("{0} {1}".format(iid, out), file=f, flush=True)

                # clear buffer
                del(instances[:])
                del(iids[:])
                del(lexelts[:])
                del(heads[:])
                cnt = 0

            token_ids, sent_lexelt, sent_iid, sent_head, _ = \
                    input_handler.tokenize_indexify(inst, ('[CLS]' if sent_attn_query else None))
            instances.append(token_ids)

            # find involved lexelts and senses
            sent_lexid = []
            for lexelt, iid in zip(sent_lexelt, sent_iid):
                lex_id = lexelt2ix.get(lexelt, 0)
                sent_lexid.append(lex_id)

            iids.append(sent_iid)
            lexelts.append(torch.tensor(sent_lexid, dtype=torch.long))
            heads.append(sent_head)
            cnt += 1

        # predict batch
        for iid, out in predict_batch(instances, iids, lexelts, heads):
            print("{0} {1}".format(iid, out), file=f, flush=True)
        f.close()
