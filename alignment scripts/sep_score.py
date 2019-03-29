#!/usr/bin/env python3

import os
import sys
import pandas
import pdb

newdir = sys.argv[2]
scored_csv = pandas.read_csv(sys.argv[1], delimiter='\t')

# (Pdb) scored_csv.keys()
# Index(['dialog', 'turn', 'glove_src_para_sim', 'glove_src_para_dist',
#        'glove_src_para_david', 'glove_src_orig_sim', 'glove_src_orig_dist',
#        'glove_src_orig_david', 'glove_orig_para_sim', 'glove_orig_para_dist',
#        'glove_orig_para_david', 'glove_align_para_sim',
#        'glove_align_para_dist', 'glove_align_para_david', 'w2v_src_para_sim',
#        'w2v_src_para_dist', 'w2v_src_para_david', 'w2v_src_orig_sim',
#        'w2v_src_orig_dist', 'w2v_src_orig_david', 'w2v_orig_para_sim',
#        'w2v_orig_para_dist', 'w2v_orig_para_david', 'w2v_align_para_sim',
#        'w2v_align_para_dist', 'w2v_align_para_david', 'elmo_src_para_sim',
#        'elmo_src_para_dist', 'elmo_src_para_david', 'elmo_src_orig_sim',
#        'elmo_src_orig_dist', 'elmo_src_orig_david', 'elmo_orig_para_sim',
#        'elmo_orig_para_dist', 'elmo_orig_para_david', 'elmo_align_para_sim',
#        'elmo_align_para_dist', 'elmo_align_para_david', 'swappable', 'swap',
#        'src', 'align', 'para', 'orig', 'label', 'response', 'cs guess',
#        'cs correct', 'color code 1', 'color code 2'],
#       dtype='object')

def normalize_and_invert(dist):
    max = 0
    min = 1
    for float_value in dist:
        if float_value > max:
            max = float_value
        elif float_value < min:
            min = float_value
    # normalize
    new_dist = []
    for old_score in dist:
        new_score = (old_score - min) / (max - min)
        # invert
        new_dist.append(1 - new_score)
    # pdb.set_trace()
    return new_dist

def normalize(dist):
    max = 0
    min = 1
    for float_value in dist:
        if float_value > max:
            max = float_value
        elif float_value < min:
            min = float_value
    # normalize
    new_dist = []
    for old_score in dist:
        new_score = (old_score - min) / (max - min)
        # invert
        new_dist.append(new_score)
    # pdb.set_trace()
    return new_dist


dialog = [x for x in scored_csv['dialog'].tolist()]
turn = [x for x in scored_csv['turn'].tolist()]
glove_src_para_sim = [x for x in scored_csv['glove_src_para_sim'].tolist()]
glove_src_para_dist = [x for x in scored_csv['glove_src_para_dist'].tolist()]
glove_src_para_david = [x for x in scored_csv['glove_src_para_david'].tolist()]
glove_src_orig_sim = [x for x in scored_csv['glove_src_orig_sim'].tolist()]
glove_src_orig_dist = [x for x in scored_csv['glove_src_orig_dist'].tolist()]
glove_src_orig_david = [x for x in scored_csv['glove_src_orig_david'].tolist()]
glove_orig_para_sim = [x for x in scored_csv['glove_orig_para_sim'].tolist()]
glove_orig_para_dist = [x for x in scored_csv['glove_orig_para_dist'].tolist()]
glove_orig_para_david = [x for x in scored_csv['glove_orig_para_david'].tolist()]
glove_align_para_sim = [x for x in scored_csv['glove_align_para_sim'].tolist()]
glove_align_para_dist = [x for x in scored_csv['glove_align_para_dist'].tolist()]
glove_align_para_david = [x for x in scored_csv['glove_align_para_david'].tolist()]
w2v_src_para_sim = [x for x in scored_csv['w2v_src_para_sim'].tolist()]
w2v_src_para_dist = [x for x in scored_csv['w2v_src_para_dist'].tolist()]
w2v_src_para_david = [x for x in scored_csv['w2v_src_para_david'].tolist()]
w2v_src_orig_sim = [x for x in scored_csv['w2v_src_orig_sim'].tolist()]
w2v_src_orig_dist = [x for x in scored_csv['w2v_src_orig_dist'].tolist()]
w2v_src_orig_david = [x for x in scored_csv['w2v_src_orig_david'].tolist()]
w2v_orig_para_sim = [x for x in scored_csv['w2v_orig_para_sim'].tolist()]
w2v_orig_para_dist = [x for x in scored_csv['w2v_orig_para_dist'].tolist()]
w2v_orig_para_david = [x for x in scored_csv['w2v_orig_para_david'].tolist()]
w2v_align_para_sim = [x for x in scored_csv['w2v_align_para_sim'].tolist()]
w2v_align_para_dist = [x for x in scored_csv['w2v_align_para_dist'].tolist()]
w2v_align_para_david = [x for x in scored_csv['w2v_align_para_david'].tolist()]
elmo_src_para_sim = [x for x in scored_csv['elmo_src_para_sim'].tolist()]
elmo_src_para_dist = [x for x in scored_csv['elmo_src_para_dist'].tolist()]
elmo_src_para_david = [x for x in scored_csv['elmo_src_para_david'].tolist()]
elmo_src_orig_sim = [x for x in scored_csv['elmo_src_orig_sim'].tolist()]
elmo_src_orig_dist = [x for x in scored_csv['elmo_src_orig_dist'].tolist()]
elmo_src_orig_david = [x for x in scored_csv['elmo_src_orig_david'].tolist()]
elmo_orig_para_sim = [x for x in scored_csv['elmo_orig_para_sim'].tolist()]
elmo_orig_para_dist = [x for x in scored_csv['elmo_orig_para_dist'].tolist()]
elmo_orig_para_david = [x for x in scored_csv['elmo_orig_para_david'].tolist()]
elmo_align_para_sim = [x for x in scored_csv['elmo_align_para_sim'].tolist()]
elmo_align_para_dist = [x for x in scored_csv['elmo_align_para_dist'].tolist()]
elmo_align_para_david = [x for x in scored_csv['elmo_align_para_david'].tolist()]
src = [x for x in scored_csv['src'].tolist()]
align = [x for x in scored_csv['align'].tolist()]
para = [x for x in scored_csv['para'].tolist()]
orig = [x for x in scored_csv['orig'].tolist()]

metrics = [(normalize(glove_src_para_sim), 'glove_src_para_sim'),
    (normalize_and_invert(glove_src_para_dist), 'glove_src_para_dist'),
    (normalize_and_invert(glove_src_para_david), 'glove_src_para_david'),
    (normalize(glove_src_orig_sim), 'glove_src_orig_sim'),
    (normalize_and_invert(glove_src_orig_dist), 'glove_src_orig_dist'),
    (normalize_and_invert(glove_src_orig_david), 'glove_src_orig_david'),
    (normalize(glove_orig_para_sim), 'glove_orig_para_sim'),
    (normalize_and_invert(glove_orig_para_dist), 'glove_orig_para_dist'),
    (normalize_and_invert(glove_orig_para_david), 'glove_orig_para_david'),
    (normalize(glove_align_para_sim), 'glove_align_para_sim'),
    (normalize_and_invert(glove_align_para_dist), 'glove_align_para_dist'),
    (normalize_and_invert(glove_align_para_david), 'glove_align_para_david'),
    (normalize(w2v_src_para_sim), 'w2v_src_para_sim'),
    (normalize_and_invert(w2v_src_para_dist), 'w2v_src_para_dist'),
    (normalize_and_invert(w2v_src_para_david), 'w2v_src_para_david'),
    (normalize(w2v_src_orig_sim), 'w2v_src_orig_sim'),
    (normalize_and_invert(w2v_src_orig_dist), 'w2v_src_orig_dist'),
    (normalize_and_invert(w2v_src_orig_david), 'w2v_src_orig_david'),
    (normalize(w2v_orig_para_sim), 'w2v_orig_para_sim'),
    (normalize_and_invert(w2v_orig_para_dist), 'w2v_orig_para_dist'),
    (normalize_and_invert(w2v_orig_para_david), 'w2v_orig_para_david'),
    (normalize(w2v_align_para_sim), 'w2v_align_para_sim'),
    (normalize_and_invert(w2v_align_para_dist), 'w2v_align_para_dist'),
    (normalize_and_invert(w2v_align_para_david), 'w2v_align_para_david'),
    (normalize(elmo_src_para_sim), 'elmo_src_para_sim'),
    (normalize_and_invert(elmo_src_para_dist), 'elmo_src_para_dist'),
    (normalize_and_invert(elmo_src_para_david), 'elmo_src_para_david'),
    (normalize(elmo_src_orig_sim), 'elmo_src_orig_sim'),
    (normalize_and_invert(elmo_src_orig_dist), 'elmo_src_orig_dist'),
    (normalize_and_invert(elmo_src_orig_david), 'elmo_src_orig_david'),
    (normalize(elmo_orig_para_sim), 'elmo_orig_para_sim'),
    (normalize_and_invert(elmo_orig_para_dist), 'elmo_orig_para_dist'),
    (normalize_and_invert(elmo_orig_para_david), 'elmo_orig_para_david'),
    (normalize(elmo_align_para_sim), 'elmo_align_para_sim'),
    (normalize_and_invert(elmo_align_para_dist), 'elmo_align_para_dist'),
    (normalize_and_invert(elmo_align_para_david), 'elmo_align_para_david')]

os.mkdir(newdir)

for values in metrics:
    scores = values[0]
    name = newdir + '/' + values[1] + '.tsv'
    outfile = open(name, 'w')
    assert(len(dialog) == len(scores))
    assert(len(turn) == len(scores))
    assert(len(para) == len(scores))
    for dial_num, turn_num, paraphrase, score in zip(dialog, turn, para, scores):
        outfile.write('\t'.join([str(dial_num), str(turn_num), paraphrase, str(score)]) + '\n')
