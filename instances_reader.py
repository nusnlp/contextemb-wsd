#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @author: chrhad
# File: instances_reader.py
# Interface to read files and load to annotations
import gzip
import io
import glob
import os
import sys
import re
import nltk
from nltk.tokenize import TreebankWordTokenizer
from unicodedata import category
import xml.etree.ElementTree as ET
from lxml import etree
from utils import open_file

class AbstractLoader:
    def __init__(self, fpath, n_surrounding=1):
        """
        Parameters
        ----------
        fpath : str
            The file path containing the XML file
        n_surrounding : int
            The number of surrounding sentences to capture context
        """
        self.sent_instances = []

    def __iter__(self):
        self._n = 0
        return self

    def __next__(self):
        if self._n < len(self.sent_instances):
            result = self.sent_instances[self._n]
            self._n += 1
            return result
        else:
            raise StopIteration

    @classmethod   
    def record_keys(cls, fpath):
        raise NotImplementedError

    @classmethod
    def write_output(cls, iid, sense):
        raise NotImplementedError


class SemEval13Loader(AbstractLoader):
    def __init__(self, fpath, n_surrounding=1):
        """
        Parameters
        ----------
        fpath : str
            The file path containing the XML file
        n_surrounding : int
            The number of surrounding sentences to capture context
        """
        super(SemEval13Loader, self).__init__(fpath, n_surrounding)
        tree = ET.parse(fpath)
        root = tree.getroot()
        assert root.tag.lower() == "corpus", "Root element must be named 'root'"
        for text in root:
            text_buffer = []  # all sentences in a text
            for sent in text:
                sent_buffer = []
                for tok in sent:
                    entries = []
                    if tok.tag == "instance":
                        lexel = tok.get("lemma") + '.' + tok.get("pos")
                        iid = tok.get("id")
                        tokspl = tok.text.split(' ')
                        entries.append((tokspl[0], lexel, iid))
                        for t in tokspl[1:]:
                            entries.append((t, '*', '*'))
                    elif tok.tag == "wf":
                        entries.append((tok.text, '#', '#'))
                    sent_buffer += entries
                text_buffer.append(sent_buffer)
            
            # create instances
            for i in range(len(text_buffer)):
                left_instance = []
                for j in range(max(i-n_surrounding, 0), i, 1):
                    left_instance += [(t[0], '#', '#') for t in text_buffer[j]] + [("[SEP]", '#', '#')]
                right_instance = []
                for j in range(i+1, min(i+n_surrounding+1, len(text_buffer)), 1):
                    right_instance += [("[SEP]", '#', '#')] + [(t[0], '#', '#') for t in text_buffer[j]]
                self.sent_instances.append([left_instance, text_buffer[i], right_instance])
    
    @classmethod
    def record_keys(cls, fpath):
        keys = {}
        with open_file(fpath, 'r') as f:
            for line in f:
                l = line.strip()
                toks = l.split(' ')
                key_str = toks[1]
                keys[toks[0]] = key_str
            f.close()
        return keys

    @classmethod
    def write_output(cls, iid, sense):
        return "{0} {1}".format(iid, sense)


class Senseval2LSLoader(AbstractLoader):
    def __init__(self, fpath, n_surrounding=1):
        """
        Parameters
        ----------
        fpath : str
            The file path containing the XML file
        n_surrounding : int
            The number of surrounding sentences to capture context
        """
        super(Senseval2LSLoader, self).__init__(fpath, n_surrounding)
        parser = etree.XMLParser(dtd_validation=True)
        tree = ET.parse(fpath, parser)
        root = tree.getroot()
        assert root.tag.lower() == "corpus", "Root element must be named 'root'"
        self.tokenizer = TreebankWordTokenizer()
        self.sent_segmenter = nltk.data.load('tokenizers/punkt/english.pickle')
        for lexelt in root:
            lexel_orig = lexelt.get("item")
            print(lexel_orig, file=sys.stderr, flush=True)
            for instance in lexelt:
                # record instance ID
                iid = instance.get("id")
                context = instance.find('context')
                sentences = []
                leftstr = context.text  # everything before the head
                headelem = context[0]  # head element
                headstr = headelem.text

                # left part of sentence: left context + head
                leftstr += headstr
                sentences += [self.remove_control_characters(s) for s in self.sent_segmenter.tokenize(leftstr.replace('\n', ' '))]
                sent_offset = len(sentences) - 1
                tok_char_end_offset = len(sentences[-1])  # offset after the last char of head
                tok_char_offset = tok_char_end_offset - len(headstr)

                sats_attr = headelem.get("sats")
                rightstr = ''
                if headelem.tail is not None:
                    rightstr += headelem.tail
                lexel_sat = None
                if sats_attr is not None:
                    lexel_sat = sats_attr.split(' ')[0].split('.')[0] + '.' + lexel.split('.')[-1]
                    for i in range(1, len(context)):
                        rightstr += context[i].text
                        rightstr += context[i].tail
                righttoks = [self.remove_control_characters(s) for s in self.sent_segmenter.tokenize(rightstr.replace('\n', ' '))]
                if len(righttoks) > 0:
                    sentences[-1] += righttoks[0]
                    sentences += righttoks[1:]

                # tokenize sentences
                tok_sentences, char2toks = zip(*[self.tokenize(sen) for sen in sentences])

                # create instances
                left_instance = []
                left_begin = max(0, sent_offset - n_surrounding) if n_surrounding >= 0 else 0
                for i in range(left_begin, sent_offset):
                    left_instance += [(t, '#', '#') for t in tok_sentences[i]] + [("[SEP]", '#', '#')]

                head_instance = []
                tok_offset = char2toks[sent_offset][tok_char_offset]
                tok_end_offset = char2toks[sent_offset][tok_char_end_offset-1]

                head_instance += [(t, '#', '#') for t in tok_sentences[sent_offset][:tok_offset]]
                lexel = lexel_sat if lexel_sat is not None else lexel_orig
                head_instance.append((tok_sentences[sent_offset][tok_offset], lexel, iid))
                head_instance += [(t, '*', '*') for t in tok_sentences[sent_offset][tok_offset+1:tok_end_offset+1]]
                head_instance += [(t, '#', '#') for t in tok_sentences[sent_offset][tok_end_offset+1:]]

                right_instance = []
                right_end = min(sent_offset + n_surrounding + 1, len(sentences))
                for i in range(sent_offset + 1, right_end):
                    right_instance += [("[SEP]", '#', '#')] + [(t, '#', '#') for t in tok_sentences[i]]

                self.sent_instances.append([left_instance, head_instance, right_instance])
    
    @classmethod
    def record_keys(cls, fpath):
        keys = {}
        with open_file(fpath, 'r') as f:
            for line in f:
                l = line.strip()
                toks = l.split(' ')
                iid = toks[1]
                key_strs = [t for t in toks[2:] if (t != 'P' and t != 'U')]
                key_str = 'U' if len(key_strs) == 0 else key_strs[0]
                keys[iid] = key_str
            f.close()
        return keys

    @classmethod
    def write_output(cls, iid, sense):
        return "{0} {1} {2}".format(iid.split('.')[0], iid, sense)

    def remove_control_characters(self, s):
        return ''.join(ch for ch in s if category(ch)[0]!='C' or category(ch) == 'Cf')

    def tokenize(self, sentence):
        tokspans = self.tokenizer.span_tokenize(sentence)
        char2tok = {}
        tokens = []
        for i, (s, e) in enumerate(tokspans):
            tokens.append(sentence[s:e])
            for j in range(s, e):
                char2tok[j] = i
        return (tokens, char2tok)

class Senseval3LSLoader(Senseval2LSLoader):
    def __init__(self, fpath, n_surrounding=1):
        """
        Parameters
        ----------
        fpath : str
            The file path containing the XML file
        n_surrounding : int
            The number of surrounding sentences to capture context
        """
        super(Senseval3LSLoader, self).__init__(fpath, n_surrounding)

    @classmethod
    def write_output(cls, iid, sense):
        return "{0} {1} {2}".format('.'.join(iid.split('.')[0:2]), iid, sense)

class SemEval13InductionLoader(AbstractLoader):
    def __init__(self, fpath, n_surrounding=1):
        """
        Parameters
        ----------
        fpath : str
            The file path containing the XML file
        n_surrounding : int
            The number of surrounding sentences to capture context
        """
        super(SemEval13InductionLoader, self).__init__(fpath, n_surrounding)
        self.tokenizer = TreebankWordTokenizer()
        self.sent_segmenter = nltk.data.load('tokenizers/punkt/english.pickle')
        fnames = sorted(glob.glob(fpath + "/*.xml"))
        for fname in fnames:
            tree = ET.parse(fname)
            root = tree.getroot()
            assert root.tag.lower() == "instances", "Root element must be named 'instances'"
            for instance in root:
                lexel = instance.get("lemma") + '.' + instance.get("partOfSpeech")
                iid = instance.get("id")
                start_offset = int(instance.get("tokenStart"))
                end_offset = int(instance.get("tokenEnd"))
                sentence = instance.text
                
                # tokenize sentences
                tok_sentence, char2tok = self.tokenize(sentence)
                
                head_instance = []
                tok_offset = char2tok[start_offset]
                tok_end_offset = char2tok[end_offset-1]

                head_instance += [(t, '#', '#') for t in tok_sentence[:tok_offset]]
                head_instance.append((tok_sentence[tok_offset], lexel, iid))
                head_instance += [(t, '*', '*') for t in tok_sentence[tok_offset+1:tok_end_offset+1]]
                head_instance += [(t, '#', '#') for t in tok_sentence[tok_end_offset+1:]]

                self.sent_instances.append([[], head_instance, []])
    
    @classmethod
    def record_keys(cls, fpath):
        keys = {}
        with open_file(fpath, 'r') as f:
            for line in f:
                l = line.strip()
                toks = l.split(' ')
                key_str = toks[1]
                keys[toks[0]] = key_str
            f.close()
        return keys

    @classmethod
    def write_output(cls, iid, sense):
        return "{0} {1}".format(iid, sense)

    def remove_control_characters(self, s):
        return ''.join(ch for ch in s if category(ch)[0]!='C' or category(ch) == 'Cf')

    def tokenize(self, sentence):
        tokspans = self.tokenizer.span_tokenize(sentence)
        char2tok = {}
        tokens = []
        for i, (s, e) in enumerate(tokspans):
            tokens.append(sentence[s:e])
            for j in range(s, e):
                char2tok[j] = i
        return (tokens, char2tok)


class InputLoaderFactory:
    def __init__(self, xml_format):
        self._xml_format = xml_format

    def load(self, train_path, num_context, key_path=None):
        InputLoader = self._get_input_loader(self._xml_format)
        inputs = InputLoader(train_path, num_context)
        keys = InputLoader.record_keys(key_path) if key_path is not None else None
        return (inputs, keys)

    def _get_input_loader(self, xml_format):
        if xml_format == 'semeval13':
            return SemEval13Loader
        elif xml_format == 'senseval2ls':
            return Senseval2LSLoader
        elif xml_format == 'senseval3ls':
            return Senseval3LSLoader
        elif xml_format == 'semeval13induction':
            return SemEval13InductionLoader
        else:
            raise ValueError(format)


if __name__ == '__main__':
    instances = SemEval13InductionLoader(sys.argv[1], 2)
    for inst in instances:
        print(inst)
