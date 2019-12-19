#!/usr/bin/env python3

import pdb
import tqdm
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn
import os
from pandas import Series, DataFrame
from pylab import rcParams
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score as avp
from sklearn import metrics 
from sklearn.metrics import classification_report

from sklearn.datasets import load_iris

labels = {}
for line in open('LogReg/labels.txt', 'r'):
    line = line.strip().split('\t')
    val = int(line[0])
    tokens = line[1]
    labels[tokens] = val

# one off:
labels["unk"] = max(labels.values()) + 1

# metrics = pd.read_csv('LogReg/multiple.cleaned.noarrays.scored.tsv', delimiter='\t')
metrics = pd.read_csv('newannots.tsv', delimiter='\t')
# test = pd.read_csv('all.tsv', delimiter='\t') #, dtype=np.str_)
test = pd.read_csv('newannots.tsv', delimiter='\t') #, dtype=np.str_)


# names = [
#     ['glove_src_para_sim', 'glove_src_orig_sim', 'glove_orig_para_sim', 'glove_align_para_sim'],
#     ['glove_src_para_dist', 'glove_src_orig_dist', 'glove_orig_para_dist', 'glove_align_para_dist'],
#     ['glove_src_para_david', 'glove_src_orig_david', 'glove_orig_para_david', 'glove_align_para_david'],
#     ['w2v_src_para_sim', 'w2v_src_orig_sim', 'w2v_orig_para_sim', 'w2v_align_para_sim'],
#     ['w2v_src_para_dist', 'w2v_src_orig_dist', 'w2v_orig_para_dist', 'w2v_align_para_dist'],
#     ['w2v_src_para_david', 'w2v_src_orig_david', 'w2v_orig_para_david', 'w2v_align_para_david'],
#     ['elmo_src_para_sim', 'elmo_src_orig_sim', 'elmo_orig_para_sim', 'elmo_align_para_sim'],
#     ['elmo_src_para_dist', 'elmo_src_orig_dist', 'elmo_orig_para_dist', 'elmo_align_para_dist'],
#     ['elmo_src_para_david', 'elmo_src_orig_david', 'elmo_orig_para_david', 'elmo_align_para_david'],
#     ['bert_src_para_sim', 'bert_src_orig_sim', 'bert_orig_para_sim', 'bert_align_para_sim'],
#     ['bert_src_para_dist', 'bert_src_orig_dist', 'bert_orig_para_dist', 'bert_align_para_dist'],
#     ['bert_src_para_david', 'bert_src_orig_david', 'bert_orig_para_david', 'bert_align_para_david'],
#     ['ng_src_para', 'ng_src_orig', 'ng_orig_para', 'ng_align_para']
#     ]
#
# results = []

def writeout(pairs, labels, outfile):
    header = '\t'.join(['index', 'sentence1', 'sentence2', 'label\n'])
    # header = '\t'.join(['index', 'sentence1', 'sentence2', 'sentence3', 'sentence4', 'label\n'])
    outfile.write(header)
    idx = 0
    for pair, label in zip(pairs, labels):
        idx += 1
        # outfile.write('\t'.join([str(idx), pair[0], pair[1], str(label) + '\n']))
        # outfile.write('\t'.join([str(idx), pair[0], pair[1], pair[2], pair[3], str(label) + '\n']))
        outfile.write('\t'.join([str(idx), " [+] ".join([pair[0], pair[1]]), " [+] ".join([pair[2], pair[3]]), str(int(label)) + '\n']))


def writeout_nolabels(pairs, outfile):
    header = '\t'.join(['index', 'sentence1', 'sentence2', 'label\n'])
    # header = '\t'.join(['index', 'sentence1', 'sentence2', 'sentence3', 'sentence4', 'label\n'])
    outfile.write(header)
    idx = 0
    for pair in pairs:
        idx += 1
        # outfile.write('\t'.join([str(idx), pair[0], pair[1], str(0) + '\n']))
        # outfile.write('\t'.join([str(idx), pair[0], pair[1], pair[2], pair[3], str(label) + '\n']))
        # try:
        outfile.write('\t'.join([str(idx), " [+] ".join([pair[0], pair[1]]), " [+] ".join([pair[2], pair[3]]), str(0) + '\n']))
        # except:
        #     pdb.set_trace()

# train_Xs = metrics[['orig', 'para']].values
train_Xs = metrics[['src', 'orig', 'align', 'para']].values
train_Ys = metrics['annotation'].values

test_Xs = test[['src', 'orig', 'align', 'para']].values
# test_Xs = test[['orig', 'para']].values
# test_Ys = test['annotation'].values

# groups = np.asarray([labels.get(x, labels['unk']) for x in metrics['label'].values])
# total = len(set(groups))
# gkf = GroupKFold(n_splits=2)
# gkf = GroupKFold(n_splits=total)



# fold_accs = []
# fold_aps = []
# fold_waps = []
#
# whole_y = []
# whole_score = []

path = 'VP'
if not os.path.exists(path):
    os.mkdir(path)


train_file = open(path + '/train.tsv', 'w')
dev_file = open(path + '/dev.tsv', 'w')

writeout(train_Xs, train_Ys, train_file)
train_file.flush()
train_file.close()
writeout_nolabels(test_Xs, dev_file)
dev_file.flush()
dev_file.close()
