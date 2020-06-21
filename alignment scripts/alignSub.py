#!/usr/bin/env python3

import sys
import gensim
import pdb
import random
import pickle as pkl
from tqdm import tqdm
from patternPara import PatternSwap
from combineGold import Combine
from build_phrase_table import PhraseTable
# from score import Score

# Usage: ./elmoEval.py elmoalignments.tsv ../data/.../batch*

class alignSub:

    def __init__(self):
        pass

    def align(self, src, tgt, indexes, phrase_table):
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

    def get_align(self, groups, phrases):
        for group in groups:
            src = group[0]
            tgt = group[1]
            idxes = group[2]
            if idxes != '':
                idxes = p.str2idx(idxes)
                phrases = p.align(src, tgt, idxes, phrases)
        return phrases

    def get_range_align(self, groups, phrases):
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

    def elmo_clean(self, elmos):
        new_elmos = []
        for line in elmos:
            if not line[2].startswith('No Align'):
                new_elmos.append(line)
        return new_elmos

    def rec_prec(self, den_dict, num_dict):
        num = 0
        den = 0
        for w1 in den_dict:
            for w2 in den_dict[w1]:
                if w1 in num_dict:
                    if w2 in num_dict:
                        num += 1
                den += 1
        return num / den

    def get_low_freq(self, corrected):
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

    def get_labels(self, corrected):
        labels = set()
        for line in corrected:
            line = line.strip().split('\t')
            if len(line) > 1:
                label = line[3]
                labels.add(label)
        return labels

    def swap(self, sents, swap_dict):
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

    def writeout(self, name, lines):
        with open(name, 'w') as of:
            for line in lines:
                # TODO we shouldn't need this check
                if line[0] != line[1]:
                    of.write('\t'.join(line) + '\n')

if __name__ == '__main__':

    ### Ready

    c = Combine()

    aS = alignSub()

    golds = []
    # for files in sys.argv[2:]:
    golds = c.read_gold(golds, sys.argv[2])

    p = PhraseTable()

    # SINGLES
    gold_singles = {}
    gold_singles = aS.get_align(golds, gold_singles)
    # pdb.set_trace()

    # HACKY PROTOTYPING
    sents = open('../data/corrected.tsv', 'r').readlines()
    labels = aS.get_labels(sents)
    low_freq = aS.get_low_freq(sents)

    # Attempt at using Sarah's code
    ps = PatternSwap()
    gold_lines = [x.split('\t') for x in open(sys.argv[2], 'r').readlines()]
    patterns = ps.extract_pattern(gold_lines)

    checked_patterns = ps.template_check(patterns)

    test_num = 0
    # range = 10
    # diffed_matches = ps.get_diff(checked_patterns, sents, test_num, until=range)
    diffed_matches = ps.get_diff(checked_patterns, sents, test_num)

    best_matches = ps.refine_matches(diffed_matches)

    phrasal_paraphrases = ps.gen_para(best_matches)

    pdb.set_trace()





    ###############################
    # OLDER STUFF---STILL NEEDED? #
    ###############################

    elmos = open(sys.argv[1], 'r').readlines()
    elmos = [x.strip().split('\t') for x in elmos]
    elmos = aS.elmo_clean(elmos)

    elmo_singles = {}
    elmo_singles = aS.get_align(elmos, elmo_singles)

    # prec = rec_prec(elmo_singles, gold_singles)
    # rec = rec_prec(gold_singles, elmo_singles)
    # f1 = 2 * ((prec * rec) / (prec + rec))


    print("single word alignments")
    # print("prec", prec)
    # print("rec", rec)
    # print("f1", f1)

    # PHRASES
    gold_phrases = {}
    gold_phrases = aS.get_range_align(golds, gold_phrases)# , use_phrase = True)

    # elmos = open(sys.argv[2], 'r').readlines()
    # elmos = [x.strip().split('\t') for x in elmos]
    # elmos = elmo_clean(elmos)

    elmo_phrases = {}
    elmo_phrases = aS.get_range_align(elmos, elmo_phrases)#  , use_phrase = True)

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



    # GEN SWAPS
    gold_sg_para = aS.swap(low_freq, gold_singles)
    elmo_sg_para = aS.swap(low_freq, elmo_singles)
    # Not currently being used
    gold_ph_para = aS.swap(low_freq, gold_phrases)
    elmo_ph_para = aS.swap(low_freq, elmo_phrases)

    # s = Score()
    # print("getting gold alignment vectors")
    # gold_sg_para = s.elmo_diffs(elmo, gold_sg_para)
    # print("getting elmo alignment vectors")
    # elmo_sg_para = s.elmo_diffs(elmo, elmo_sg_para)

    aS.writeout('gold_singular_swap.tsv', gold_sg_para)
    aS.writeout('gold_phrase_swap.tsv', gold_ph_para)
    aS.writeout('elmo_singular_swap.tsv', elmo_sg_para)


    # pdb.set_trace()

