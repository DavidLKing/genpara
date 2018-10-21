#!/usr/bin/env python3

import sys
import gensim
import pdb
from allennlp.modules.elmo import Elmo, batch_to_ids
from combineGold import Combine
from build_phrase_table import PhraseTable
from score import Score

# Usage: ./elmoEval.py elmoalignments.tsv ../data/.../batch*

def get_align(groups, phrases, use_phrase = False):
    for group in groups:
        src = group[0]
        tgt = group[1]
        idxes = group[2]
        if idxes != '':
            idxes = p.str2idx(idxes)
            if use_phrase:
                idxes = p.conv2range(idxes)
            else:
                # get analgous output to above
                idxes = [[[i[0]], [i[1]]] for i in idxes]
            phrases = p.align(src, tgt, idxes, phrases)
    # pdb.set_trace()
    return phrases

# def get_range_align(groups, phrases):
#     for group in groups:
#         src = group[0]
#         tgt = group[1]
#         idxes = group[2]
#         if idxes != '':
#             idxes = p.str2idx(idxes)
#             phrases = p.align_phrase(src, tgt, idxes, phrases)
#     return phrases

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

def get_low_freq(corrected):
    labels = {}
    sents = []
    for line in corrected:
        line = line.strip().split('\t')
        if len(line) > 1:
            label = line[3]
            if label not in labels:
                labels[label] = 0
            labels[label] += 1
            sents.append(line)
    return_sents = []
    for line in sents:
        if labels[line[3]] < 20:
            return_sents.append(line)
    return return_sents

def swap(sents, swap_dict):
    paraphrases = []
    mltplsrc = 0
    total = 0
    sum_multiple = 0
    sum_all = 0
    for line in sents:
        total += 1
        sent = line[0]
        for swappable in swap_dict:
            # TODO temp hack to stop history + i > we = hwestory
            if ' ' + swappable + ' ' in sent:
                for swap in swap_dict[swappable]:
                    para = sent.replace(swappable, swap)
                    paraphrases.append([para] + line)
                    srcs = len(swap_dict[swappable][swap]['src_sents'])
                    if srcs > 1:
                        mltplsrc += 1
                        sum_multiple += srcs
                    sum_all += srcs
    print("multiple sources", mltplsrc, total, mltplsrc / total)
    print("avg multiple source count", sum_multiple, total, sum_multiple / total)
    print("overall average source count", sum_all, total, sum_all / total)
    return paraphrases

def writeout(name, lines):
    with open(name, 'w') as of:
        for line in lines:
            # TODO we shouldn't need this check
            if line[0] != line[1]:
                of.write('\t'.join(line) + '\n')

### ELLMO ###
options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

elmo = Elmo(options_file, weight_file, 2, dropout=0)
# messy prototype
single_sent_elmo = lambda sent: elmo(batch_to_ids([sent]))
### Ready
pdb.set_trace()

c = Combine()

golds = []
for files in sys.argv[2:]:
    golds = c.read_gold(golds, files)

p = PhraseTable()

# SINGLES
gold_singles = {}
gold_singles = get_align(golds, gold_singles)

elmos = open(sys.argv[1], 'r').readlines()
elmos = [x.strip().split('\t') for x in elmos]
elmos = elmo_clean(elmos)

elmo_singles = {}
elmo_singles = get_align(elmos, elmo_singles)

prec = rec_prec(elmo_singles, gold_singles)
rec = rec_prec(gold_singles, elmo_singles)
f1 = 2 * ((prec * rec) / (prec + rec))

print("single word alignments")
print("prec", prec)
print("rec", rec)
print("f1", f1)

# PHRASES
gold_phrases = {}
gold_phrases = get_align(golds, gold_phrases, use_phrase = True)

elmos = open(sys.argv[1], 'r').readlines()
elmos = [x.strip().split('\t') for x in elmos]
elmos = elmo_clean(elmos)

elmo_phrases = {}
elmo_phrases = get_align(elmos, elmo_phrases, use_phrase = True)

prec = rec_prec(elmo_phrases, gold_phrases)
rec = rec_prec(gold_phrases, elmo_phrases)
f1 = 2 * ((prec * rec) / (prec + rec))

print("phrasal alignments")
print("prec", prec)
print("rec", rec)
print("f1", f1)

# HACKY PROTOTYPING
sents = open('../data/corrected.tsv', 'r').readlines()
low_freq = get_low_freq(sents)

# GEN SWAPS
gold_sg_para = swap(low_freq, gold_singles)
elmo_sg_para = swap(low_freq, elmo_singles)
# Not currently being used
gold_ph_para = swap(low_freq, gold_phrases)
elmo_ph_para = swap(low_freq, elmo_phrases)

writeout('gold_singular_swap.tsv', gold_sg_para)
writeout('elmo_singular_swap.tsv', elmo_sg_para)


# pdb.set_trace()
