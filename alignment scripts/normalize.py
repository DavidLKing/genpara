#!/usr/bin/env python3

import os
import sys
import pandas
import pdb

# USAGE:
# ./normalize.py scored.py newdir scored.tsv

newdir = sys.argv[2]
scored_csv = pandas.read_csv(sys.argv[1], delimiter='\t')
# scored_csv = pandas.read_csv(open(sys.argv[1], 'rU'), encoding='utf-8', engine='c', delimiter='\t')
# pd.read_csv(open('test.csv','rU'), encoding='utf-8', engine='c')

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


# dialog = [x for x in scored_csv['dialog'].tolist()]
# turn = [x for x in scored_csv['turn'].tolist()]
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
elmo_sims = scored_csv['elmo_src_para_sim'] + scored_csv['elmo_src_orig_sim'] + scored_csv['elmo_orig_para_sim'] + scored_csv['elmo_align_para_sim']
elmo_sims = elmo_sims.tolist()
elmo_dist = scored_csv['elmo_src_para_dist'] + scored_csv['elmo_src_orig_dist'] + scored_csv['elmo_orig_para_dist'] + scored_csv['elmo_align_para_dist']
elmo_dist = elmo_dist.tolist()
elmo_david = scored_csv['elmo_src_para_david'] + scored_csv['elmo_src_orig_david'] + scored_csv['elmo_orig_para_david'] + scored_csv['elmo_align_para_david']
elmo_david = elmo_david.tolist()
ng_src_para = [x for x in scored_csv['ng_src_para'].tolist()]
ng_src_orig = [x for x in scored_csv['ng_src_orig'].tolist()]
ng_orig_para = [x for x in scored_csv['ng_orig_para'].tolist()]
ng_align_para = [x for x in scored_csv['ng_align_para'].tolist()]
ng_sum = scored_csv['ng_src_para'] + scored_csv['ng_src_orig'] + scored_csv['ng_orig_para'] + scored_csv['ng_align_para']
ng_sum = ng_sum.tolist()
swappable = [x for x in scored_csv['swappable'].tolist()]
swap = [x for x in scored_csv['swap'].tolist()]
src = [x for x in scored_csv['src'].tolist()]
align = [x for x in scored_csv['align'].tolist()]
para = [x for x in scored_csv['para'].tolist()]
orig = [x for x in scored_csv['orig'].tolist()]
# label = [x for x in scored_csv['label']]
annotations = [x for x in scored_csv['annotations']]

# pdb.set_trace()

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
    (normalize_and_invert(elmo_align_para_david), 'elmo_align_para_david'),
    (normalize(elmo_sims), 'elmo_sims'),
    (normalize_and_invert(elmo_dist), 'elmo_dist'),
    (normalize_and_invert(elmo_david), 'elmo_joint'),
    (normalize_and_invert(ng_src_para), 'ng_src_para'),
    (normalize_and_invert(ng_src_orig), 'ng_src_orig'),
    (normalize_and_invert(ng_orig_para), 'ng_orig_para'),
    (normalize_and_invert(ng_align_para), 'ng_align_para'),
    (normalize_and_invert(ng_sum), 'ng_sum'),
    (swappable, 'swappable'),
    (swap, 'swap'),
    (src, 'src'),
    (align, 'align'),
    (orig, 'orig'),
    (para, 'para'),
    # (label, 'label'),
    (annotations, 'annotations')]


if not os.path.isdir(newdir):
    os.mkdir(newdir)

header = [x[1] for x in metrics]
scores = [x[0] for x in metrics]

pdb.set_trace()

outfile = open(newdir + '/scored.tsv', 'w')

outfile.write('\t'.join(header) + '\n')

# pdb.set_trace()
for values in zip(*scores):
    # pdb.set_trace()
    outfile.write('\t'.join([str(x) for x in values]) + '\n')

