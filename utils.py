#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @author: chrhad
# File: utils.py
import gzip

# Open file descriptor based on extension
def open_file(fname, mode='rt', encoding='utf-8'):
    if mode == 'r' or mode == 'w':
        mode += 't'
    if mode.endswith('b'):
        f = gzip.open(fname, mode) if fname.endswith('.gz') \
                else open(fname, mode)
        return f
    else:
        f = gzip.open(fname, mode, encoding=encoding) if fname.endswith('.gz') \
                else open(fname, mode, encoding=encoding)
        return f
