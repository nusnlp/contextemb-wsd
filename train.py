# /usr/bin/env python
# -*- coding: utf-8 -*-
# File: train.py
# @author: chrhad
# Train BERT classifier for all words WSD
import argparse
import glob
import io
import random
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertAdam
from bert_input_handler import BertInputHandler
from instances_reader import InputLoaderFactory
from utils import open_file
from model import BertSenseClassifier, pad

def split_tuple_list(tuplist):
    return tuple(map(lambda i: list(map(lambda x:x[i], tuplist)), range(len(tuplist[0]))))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
            "Train word sense disambiguation model for one lexelt")
    argparser.add_argument('train_path', help="Training instance file in XML format")
    argparser.add_argument('key_path', help="Training key file")
    argparser.add_argument('model_dir', help="Directory to output model")

    # Optional arguments
    argparser.add_argument('--xml-format', choices=['semeval13', 'senseval2ls', 'senseval3ls'],
            default='semeval13', help='Input XML file format (default: semeval13)')
    argparser.add_argument('--dev-ratio', type=float, default=0.2,
            help="ratio of development set to take from the training set (default: 0.2)")
    argparser.add_argument('--devset-path', help='File path to the development data')
    argparser.add_argument('--devkey-path', help='File path to the development key')
    argparser.add_argument('--bert-model', default="bert-base-cased",
            help="BERT pre-trained model to use")
    argparser.add_argument('--num-context', type=int, default=1,
            help="Number of sentences to the left and right each (default: 1)")
    argparser.add_argument('--layer', type=int, default=11,
            help="BERT layer for word representation (default: 11 = top layer of bert-base-cased)")
    argparser.add_argument('--use-glu', action='store_true',
            help="Use gated linear unit on the word by the sentence representation (default: not used)")
    argparser.add_argument('--residual-glu', action='store_true',
            help="Add the original BERT output to the GLU output (default: no)")
    argparser.add_argument('--top-attn-head', type=int, default=1,
            help="Number of attention head before FFNN prediction (default: 1)")
    argparser.add_argument('--sent-attn-query', action='store_true',
            help="Use attention query from sentence vector instead of a common random variable (default: random)")
    argparser.add_argument('--optimizer', choices=['sgd', 'adam', 'bert-adam'], default='bert-adam',
            help="Gradient-based training algorithm (default: bert-adam)")
    argparser.add_argument('--lr', type=float, default=1e-3,
            help="Learning rate (default: 1e-3)")
    argparser.add_argument('--dropout', type=float, default=0,
            help="Dropout rate for top layer FFNN (default: 0)")
    argparser.add_argument('--attn-dropout', type=float, default=0.,
            help="Dropout rate for top layer attention (default: 0)")
    argparser.add_argument('--num-epochs', type=int, default=50,
            help="Number of training epochs (default: 50)")
    argparser.add_argument('--batch-size', type=int, default=16,
            help="Training batch size (default: 16)")
    argparser.add_argument('--dev-batch-size', type=int, default=16,
            help="Development batch size (default: 16)")
    argparser.add_argument('--no-shuffle', action='store_true', default=False,
            help="Do not shuffle training data (default: shuffle)")
    argparser.add_argument('--patience', type=int, default=10,
            help="Number of epochs after best development set result does not improve (default: 10)")
    argparser.add_argument('--iter-checkpoint', type=int, default=1000,
            help="Number of iterations to show training progress outside epoch (default: 1000)")
    argparser.add_argument('--seed', type=int, default=123, help="Random seed (default: 123)")
    args = argparser.parse_args()
    
    torch.manual_seed(args.seed)  # set fixed random seed
    torch.cuda.manual_seed(args.seed)  # set fixed random seed
    torch.backends.cudnn.deterministic=True

    mdir = args.model_dir  # path to store trained model
    if not os.path.exists(mdir):
        os.mkdir(mdir)
    elif not os.path.isdir(mdir):
        sys.exit("{0} exists but not a directory".format(mdir))

    # 0. Load keys and instances, assign indices
    print("| Loding BERT tokenizer from {0}".format(args.bert_model),
            file=sys.stderr, flush=True)
    input_handler = BertInputHandler(args.bert_model)
    pad_ix = input_handler.pad_idx()

    print("| Loading training instances from {0} with format of {1}".format(args.train_path, args.xml_format),
            file=sys.stderr, flush=True)
    input_loader = InputLoaderFactory(args.xml_format)
    train_sentences, train_keys = input_loader.load(args.train_path, args.num_context, args.key_path)
    
    dev_sentences, dev_keys = (None, None)
    if args.devset_path is not None:
        print("| Loading development instances from {0}".format(args.devset_path), file=sys.stderr, flush=True)
        dev_sentences, dev_keys = input_loader.load(args.devset_path, args.num_context, args.devkey_path)
    
    lexelt2ix = {'unk.UNK': 0}
    ix2lexelt = ['unk.UNK']
    key2ix = {'U': 0}
    ix2key = ['U']
    key2lexelt = [0]
    
    train_instances = []
    train_iids = []
    train_lexelts = []
    train_heads = []
    train_senses = []
    for inst in train_sentences:
        token_ids, sent_lexelt, sent_iid, sent_head, _ = \
                input_handler.tokenize_indexify(inst, ('[CLS]' if args.sent_attn_query else None))
        train_instances.append(token_ids)

        # find or record index for involved lexelts and senses
        sent_senses = []
        sent_lexid = []
        for lexelt, iid in zip(sent_lexelt, sent_iid):
            lex_id = lexelt2ix.setdefault(lexelt, len(ix2lexelt))
            sent_lexid.append(lex_id)
            sense = train_keys.get(iid, 'U')
            if lex_id == len(ix2lexelt):
                ix2lexelt.append(lexelt)
            sense_id = key2ix.setdefault(sense, len(ix2key))
            sent_senses.append(sense_id)
            if sense_id == len(ix2key):
                ix2key.append(sense)
                key2lexelt.append(lex_id)

        train_iids.append(sent_iid)
        train_lexelts.append(torch.tensor(sent_lexid, dtype=torch.long))
        train_heads.append(sent_head)
        train_senses.append(torch.tensor(sent_senses, dtype=torch.long))
    train_len = len(train_instances)
    unk_sensidx = 0

    torch.save(lexelt2ix, "{0}/lexelt_idx.bin".format(mdir))
    torch.save(ix2key, "{0}/sense_idx.bin".format(mdir))
    torch.save(key2lexelt, "{0}/sense_lexelt".format(mdir))
    torch.save(args, "{0}/params.bin".format(mdir))

    # 0.2 Developement data
    dev_instances = []
    dev_iids = []
    dev_lexelts = []
    dev_heads = []
    dev_senses = []
    if dev_sentences is not None:
        for inst in dev_sentences:
            token_ids, sent_lexelt, sent_iid, sent_head, _ = \
                    input_handler.tokenize_indexify(inst, ('[CLS]' if args.sent_attn_query else None))
            dev_instances.append(token_ids)

            # find involved lexelts and senses
            sent_senses = []
            sent_lexid = []
            for lexelt, iid in zip(sent_lexelt, sent_iid):
                lex_id = lexelt2ix.get(lexelt, 0)
                sense = dev_keys.get(iid, 'U')
                sense_id = key2ix.get(sense, key2ix['U'])
                sent_lexid.append(lex_id)
                sent_senses.append(sense_id)

            dev_iids.append(sent_iid)
            dev_lexelts.append(torch.tensor(sent_lexid, dtype=torch.long))
            dev_heads.append(sent_head)
            dev_senses.append(torch.tensor(sent_senses, dtype=torch.long))
    elif args.dev_ratio > 0.:
        all_X = list(zip(train_instances, train_iids, train_lexelts, train_heads))
        train_X, dev_X, train_y, dev_senses = train_test_split(
                all_X, train_senses, test_size=args.dev_ratio, random_state=args.seed)

        # repopulate train_instances
        train_instances, train_iids, train_lexelts, train_heads = split_tuple_list(train_X)
        dev_instances, dev_iids, dev_lexelts, dev_heads = split_tuple_list(dev_X)
        train_senses = train_y
        train_len = len(train_instances)
    dev_len = len(dev_instances)
    
    # 2. Construct model and define loss function
    print("| Building network architecture for {1:d} lexical items with BERT pretrained model: {0}".format(
        args.bert_model, len(lexelt2ix)), file=sys.stderr, flush=True)
    model = BertSenseClassifier(args.bert_model, len(ix2key), key2lexelt,
            mlp_dropout=args.dropout, attn_dropout=args.attn_dropout,
            layer=args.layer, use_glu=args.use_glu, residual_glu=args.residual_glu,
            top_attn_head=args.top_attn_head, sent_attn_query=args.sent_attn_query)
    model.cuda()

    # loss function definition
    print("| Defining optimization with the algorithm: {0}".format(args.optimizer),
            file=sys.stderr, flush=True)
    loss_fn = nn.NLLLoss(ignore_index=unk_sensidx)

    # optimizer definition
    optimizer = None
    if args.optimizer == 'bert-adam':
        optimizer = BertAdam(model.train_parameters(), lr=args.lr)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.train_parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.train_parameters(), lr=args.lr)

    # 3. Train model (shuffle lexelt list, then shuffle each item inside)
    bsz = args.batch_size
    dbsz = args.dev_batch_size
    best_error = float('inf')
    stalled = 0
    model_fname = "{0}/model.bin".format(mdir)
    iter_chk = args.iter_checkpoint
    patience = args.patience
    for epoch in range(args.num_epochs):
        print("| Epoch {0:3d} started.".format(epoch+1), file=sys.stderr, flush=True)
        model.train()
        train_loss = 0.
        num_iter = 0
        num_inst = 0
        # Shuffle training instances to prevent overfitting, first the lexelts
        if not args.no_shuffle:
            random.seed(args.seed * (epoch+1))  # reset random seed at each epoch, useful if training is restartable
            train_lex_tuples = list(zip(train_instances, train_iids, train_heads, train_lexelts, train_senses))
            random.shuffle(train_lex_tuples)
            train_instances, train_iids, train_heads, train_lexelts, train_senses = \
                    split_tuple_list(train_lex_tuples)

        for bstart in range(0, train_len, bsz):
            model.zero_grad()
            bend = min(bstart + bsz, train_len)

            batch_sents = pad(train_instances[bstart:bend], pad_ix)
            batch_heads = train_heads[bstart:bend]
            batch_lexelts = train_lexelts[bstart:bend]
            batch_senses = train_senses[bstart:bend]

            batch_sents = batch_sents.cuda()
            batch_heads = [x.cuda() for x in batch_heads]
            batch_lexelts = [x.cuda() for x in batch_lexelts]
            batch_senses = torch.cat([x.cuda() for x in batch_senses])

            # Compute log-probabilities of each sense
            lprobs = model(batch_sents, batch_heads, batch_lexelts)
            loss = loss_fn(lprobs, batch_senses)
            loss.backward()  # compute gradient
            optimizer.step() # update parameters
            loss_val = loss.item()  # keep track of loss value, to print at the epoch end

            # record statistics
            train_loss += loss_val * batch_senses.size(0)
            num_iter += 1
            num_inst += batch_senses.size(0)
            if num_iter % iter_chk == 0:
                print("| Epoch {0:3d} iter {1:d} training loss = {2:.4f}.".format(
                    epoch+1, num_iter, loss_val), file=sys.stderr,
                    flush=True)
        
        train_loss /= num_inst

        # Measure on development data (seen items only)
        with torch.no_grad():
            model.eval()
            errcnt = 0
            instcnt = 0
            for bstart in range(0, dev_len, dbsz):
                bend = min(bstart + dbsz, dev_len)

                batch_sents = pad(dev_instances[bstart:bend], pad_ix)
                batch_heads = dev_heads[bstart:bend]
                batch_lexelts = dev_lexelts[bstart:bend]
                batch_senses = dev_senses[bstart:bend]

                batch_sents = batch_sents.cuda()
                batch_heads = [x.cuda() for x in batch_heads]
                batch_lexelts = [x.cuda() for x in batch_lexelts]
                batch_senses = torch.cat([x.cuda() for x in batch_senses])

                # Compute model prediction
                lprobs = model(batch_sents, batch_heads, batch_lexelts)
                _ , argmax = lprobs.max(dim=-1)
                batch_errcnt = argmax.ne(batch_senses).sum().item()
                errcnt += batch_errcnt
                instcnt += sum([len(x) for x in batch_heads])
            error_rate = errcnt / instcnt if instcnt > 0 else 0
            is_best = (error_rate == 0 or error_rate < best_error)
            best_str = ""
            if is_best:
                stalled = 0
                best_error = error_rate
                torch.save(model.state_dict(), model_fname)
                best_str += " new best"
            else:
                stalled += 1

            print("| Epoch {0:3d} final training loss = {1:.4f}, dev error = {2:.4f}{3}".format(
                epoch+1, train_loss, error_rate, best_str), file=sys.stderr, flush=True)

        if stalled >= patience:
            print("| Early stopping, no further improvement.", file=sys.stderr, flush=True)
            break
