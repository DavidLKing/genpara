#!/usr/bin/env python3

import pdb
import tqdm
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn

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
for line in open('labels.txt', 'r'):
    line = line.strip().split('\t')
    val = int(line[0])
    tokens = line[1]
    labels[tokens] = val

# one off:
labels["unk"] = max(labels.values()) + 1

metrics = pd.read_csv('multiple.cleaned.noarrays.scored.tsv', delimiter='\t')


names = [
    ['glove_src_para_sim', 'glove_src_orig_sim', 'glove_orig_para_sim', 'glove_align_para_sim'],
    ['glove_src_para_dist', 'glove_src_orig_dist', 'glove_orig_para_dist', 'glove_align_para_dist'],
    ['glove_src_para_david', 'glove_src_orig_david', 'glove_orig_para_david', 'glove_align_para_david'],
    ['w2v_src_para_sim', 'w2v_src_orig_sim', 'w2v_orig_para_sim', 'w2v_align_para_sim'],
    ['w2v_src_para_dist', 'w2v_src_orig_dist', 'w2v_orig_para_dist', 'w2v_align_para_dist'],
    ['w2v_src_para_david', 'w2v_src_orig_david', 'w2v_orig_para_david', 'w2v_align_para_david'],
    ['elmo_src_para_sim', 'elmo_src_orig_sim', 'elmo_orig_para_sim', 'elmo_align_para_sim'],
    ['elmo_src_para_dist', 'elmo_src_orig_dist', 'elmo_orig_para_dist', 'elmo_align_para_dist'],
    ['elmo_src_para_david', 'elmo_src_orig_david', 'elmo_orig_para_david', 'elmo_align_para_david'],
    ['bert_src_para_sim', 'bert_src_orig_sim', 'bert_orig_para_sim', 'bert_align_para_sim'],
    ['bert_src_para_dist', 'bert_src_orig_dist', 'bert_orig_para_dist', 'bert_align_para_dist'],
    ['bert_src_para_david', 'bert_src_orig_david', 'bert_orig_para_david', 'bert_align_para_david'],
    ['ng_src_para', 'ng_src_orig', 'ng_orig_para', 'ng_align_para']
    ]

results = []

for nam in tqdm.tqdm(names):
    embed = nam[0].split('_')
    embed_type = embed[0]
    composition = embed[-1].replace('david', 'joint').replace('para', 'n/a')

    X = metrics[nam].values
    y = metrics['annotation'].values
    groups = np.asarray([labels.get(x, labels['unk']) for x in metrics['label'].values])
    # gkf = GroupKFold(n_splits=2)
    total = len(set(groups))
    gkf = GroupKFold(n_splits=total)

    fold_accs = []
    fold_aps = []
    fold_waps = []

    whole_y = []
    whole_score = []

    for train, test in gkf.split(X, y, groups=groups):
        # print("Currently on {}".format(current))
        clf = LogisticRegression(random_state=0, solver='lbfgs', n_jobs=32, verbose=0, max_iter=1000)
        train_X = np.asarray([X[x] for x in train])
        train_y = np.asarray([y[x] for x in train])
        test_X = np.asarray([X[x] for x in test])
        test_y = np.asarray([y[x] for x in test])
        clf.fit(train_X, train_y)
        probas = clf.predict_proba(test_X)
        is_para_prob = [x[1] for x in probas]
        is_not_para_prob = [x[1] for x in probas]
        whole_y += test_y.tolist()
        whole_score += is_para_prob
        if 1 in test_y and 0 in test_y:
            fold_aps.append(avp(test_y, is_para_prob))
            avg_prec = avp(test_y, is_para_prob)
            fold_aps.append(avg_prec)
            weight = test_y.shape[0] / y.shape[0]
            fold_waps.append(avg_prec * weight)
        fold_accs.append(clf.score(test_X, test_y))
        # current += 1

    whole_ap = avp(whole_y, whole_score)
    results.append((embed_type,
                    composition,
                    str(np.mean(fold_accs)),
                    str(whole_ap),
                    str(np.mean(fold_aps)),
                    str(sum(fold_waps))))
    # print("fold_aps:", fold_apsn)

outfile = open('acc_avp_and_map.tsv', 'w')

print('\t'.join(['embed', 'comp', 'acc', 'aveP', 'map', '% * map']))
outfile.write('\t'.join(['embed', 'comp', 'acc', 'aveP', 'map', '% * map']) + '\n')
for res in results:
    print('\t'.join(res))
    outfile.write('\t'.join(res) + '\n')



