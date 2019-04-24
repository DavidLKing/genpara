#!/usr/bin/env python3

import sys
import gensim
import pdb
import random
import pickle as pkl
from combineGold import Combine
from build_phrase_table import PhraseTable
# from score import Score

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
        # todo, how to clean?
        # thought:
        # allidxes = group[2].split(' ')
        # for idx in allidxes:
        #     for other_idx in allidxes:
        #         if type(idx) != list:
        #             idx = idx.split('-')
        #         if type(other_idx) != list:
        #             other_idx = other_idx.split('-')
        #         if idx[0] == other_idx[0] or idx[1] == other_idx[0]:
        #             if idx != other_idx:
        #                 allidxes.pop(allidxes.index('-'.join(idx)))
        # idxes = ' '.join(allidxes)
        # pdb.set_trace()
        # pdb.set_trace()
        if idxes != '':
            idxes = p.str2idx(idxes)
            phrases = p.align(src, tgt, idxes, phrases)
    return phrases

def get_range_align(groups, phrases):
    for group in groups:
        src = group[0]
        tgt = group[1]
        idxes = group[2]
        if idxes != '':
            idxes = p.str2idx(idxes)
            # if use_phrase:
            idxes = p.conv2range(idxes)
            # else:
            #     # get analgous output to above
            #     idxes = [[[i[0]], [i[1]]] for i in idxes]
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
    header = [
        'swappable',
        'swap',
        'src',
        'align',
        'para',
        'orig',
        'label',
        'response',
        'cs guess',
        'cs correct',
        'color code 1',
        'color code 2'
    ]
    paraphrases.append(header)
    for line in sents:
        # pdb.set_trace()
        total += 1
        sent = line[0]
        for swappable in swap_dict:
            # TODO temp hack to stop history + i > we = hwestory
            # TODO there has to be a better way to do this: tokenization?
            if ' ' + swappable + ' ' in sent:
                for swap in swap_dict[swappable]:
                    src = random.choice(swap_dict[swappable][swap]['src'])
                    align = random.choice(swap_dict[swappable][swap]['align'])
                    para = sent.replace(swappable, swap)
                    # pdb.set_trace()
                    new_array = [swappable, swap, src, align, para] + line
                    assert(len(new_array) == len(header))
                    paraphrases.append(new_array)
    return paraphrases

def writeout(name, lines):
    with open(name, 'w') as of:
        for line in lines:
            # TODO we shouldn't need this check
            if line[0] != line[1]:
                of.write('\t'.join(line) + '\n')






### Ready

c = Combine()

golds = []
for files in sys.argv[2:]:
    golds = c.read_gold(golds, files)

p = PhraseTable()

# SINGLES
gold_singles = {}
gold_singles = get_align(golds, gold_singles)
# pdb.set_trace()

elmos = open(sys.argv[1], 'r').readlines()
elmos = [x.strip().split('\t') for x in elmos]
elmos = elmo_clean(elmos)

# elmo_singles = {}
# elmo_singles = get_align(elmos, elmo_singles)

# prec = rec_prec(elmo_singles, gold_singles)
# rec = rec_prec(gold_singles, elmo_singles)
# f1 = 2 * ((prec * rec) / (prec + rec))


print("single word alignments")
# print("prec", prec)
# print("rec", rec)
# print("f1", f1)

# PHRASES
gold_phrases = {}
gold_phrases = get_range_align(golds, gold_phrases)# , use_phrase = True)

# elmos = open(sys.argv[2], 'r').readlines()
# elmos = [x.strip().split('\t') for x in elmos]
# elmos = elmo_clean(elmos)

# elmo_phrases = {}
# elmo_phrases = get_range_align(elmos, elmo_phrases)#  , use_phrase = True)

# save them
pkl.dump(gold_singles, open('gold_singles.pkl', 'wb'))
# pkl.dump(elmo_singles, open('elmo_singles.pkl', 'wb'))
pkl.dump(gold_phrases, open('gold_phrases.pkl', 'wb'))
# pkl.dump(elmo_phrases, open('elmo_phrases.pkl', 'wb'))

# prec = rec_prec(elmo_phrases, gold_phrases)
# rec = rec_prec(gold_phrases, elmo_phrases)
# f1 = 2 * ((prec * rec) / (prec + rec))

print("phrasal alignments")
# print("prec", prec)
# print("rec", rec)
# print("f1", f1)

# HACKY PROTOTYPING
sents = open('../data/corrected.tsv', 'r').readlines()
low_freq = get_low_freq(sents)

# GEN SWAPS
gold_sg_para = swap(low_freq, gold_singles)
# elmo_sg_para = swap(low_freq, elmo_singles)
# Not currently being used
gold_ph_para = swap(low_freq, gold_phrases)
# elmo_ph_para = swap(low_freq, elmo_phrases)

# s = Score()
# print("getting gold alignment vectors")
# gold_sg_para = s.elmo_diffs(elmo, gold_sg_para)
# print("getting elmo alignment vectors")
# elmo_sg_para = s.elmo_diffs(elmo, elmo_sg_para)

writeout('gold_singular_swap.tsv', gold_sg_para)
writeout('gold_phrase_swap.tsv', gold_ph_para)
# writeout('elmo_singular_swap.tsv', elmo_sg_para)


# pdb.set_trace()

