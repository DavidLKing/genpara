#!/usr/bin/env python3

import sys
import pdb
from combineGold import Combine
from build_phrase_table import PhraseTable


# Usage: ./elmoEval.py elmoalignments.tsv ../data/.../batch*

def align(src, tgt, indexes, phrase_table):
    # mocked up from PhraseTable.py to not create phrases
    src = src.split()
    tgt = tgt.split()
    # indexes = p.conv2range(indexes)
    for pair in indexes:
        # src_phrase = self.gen_phrase(src, pair[0])
        # tgt_phrase = self.gen_phrase(tgt, pair[1])
        src_phrase = src[pair[0]]
        tgt_phrase = tgt[pair[1]]
        if src_phrase != tgt_phrase:
            if src_phrase not in phrase_table:
                phrase_table[src_phrase] = []
            if tgt_phrase not in phrase_table[src_phrase]:
                phrase_table[src_phrase].append(tgt_phrase)
    return phrase_table

def get_align(groups, phrases):
    for group in groups:
        src = group[0]
        tgt = group[1]
        idxes = group[2]
        if idxes != '':
            idxes = p.str2idx(idxes)
            phrases = align(src, tgt, idxes, phrases)
    return phrases

def elmo_clean(elmos):
    new_elmos = []
    for line in elmos:
        if not line[2].startswith('No Align'):
            new_elmos.append(line)
    return new_elmos

def rec_prec(den_dict, num_dict):
    num = 0
    den = 0
    for w1 in den_dict:
        for w2 in den_dict[w1]:
            if w1 in num_dict:
                if w2 in num_dict:
                    num += 1
            den += 1
    return num / den

c = Combine()

golds = []
for files in sys.argv[2:]:
    golds = c.read_gold(golds, files)

p = PhraseTable()
gold_phrases = {}
gold_phrases = get_align(golds, gold_phrases)


elmos = open(sys.argv[1], 'r').readlines()
elmos = [x.strip().split('\t') for x in elmos]
elmos = elmo_clean(elmos)

elmo_phrases = {}
elmo_phrases = get_align(elmos, elmo_phrases)

prec = rec_prec(elmo_phrases, gold_phrases)
rec = rec_prec(gold_phrases, elmo_phrases)
f1 = 2 * ((prec * rec) / (prec + rec))

print("prec", prec)
print("rec", rec)
print("f1", f1)

pdb.set_trace()
