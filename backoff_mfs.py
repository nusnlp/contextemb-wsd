#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: backoff_mfs.py
# @author: chrhad
# Given neural-tagged WSD output and MFS output, back-off to MFS output if the former is unknown
import argparse
import io
import sys
import os

from instances_reader import open_file

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
            "Run most-frequent sense backoff if the prediction is unknown")
    argparser.add_argument('pred_path', help="Output prediction by neural system")
    argparser.add_argument('mfs_path', help="Output prediction by MFS system")
    args = argparser.parse_args()
    
    mfs_senses = {}
    with open_file(args.mfs_path, 'r') as f:
        for line in f:
            l = line.strip()
            toks = l.split()
            mfs_senses[toks[0]] = toks[1]
        f.close()

    with open_file(args.pred_path, 'r') as f:
        for line in f:
            l = line.strip()
            toks = l.split()
            iid = toks[0]
            sense = mfs_senses.get(iid, 'U') if toks[1] == 'U' else toks[1]
            print("{0} {1}".format(iid, sense))
        f.close()
