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

from statsmodels.stats.contingency_tables import mcnemar


def signif(corrs1, corrs2):
    bothright = 0
    YesNo = 0
    NoYes = 0
    bothwrong = 0



    statistic = lambda x,y: (x - y)**2 / (x + y)

    for c1, c2 in zip(corrs1, corrs2):
        if c1 == True and c2 == True:
            bothright += 1
        elif c1 == True and c2 == False:
            YesNo += 1
        elif c1 == False and c2 == True:
            NoYes += 1
        elif c1 == False and c2 == False:
            bothwrong += 1
        else:
            print("Error")
            pdb.set_trace()
    
    # print("manual:", statistic(YesNo, NoYes))
    # print("statsmodels", mcnemar(np.array([[bothright, YesNo], [NoYes, bothwrong]])))
    # print("statsmodels exact false", mcnemar(np.array([[bothright, YesNo], [NoYes, bothwrong]]), exact=False))
    # print("statsmodels correction false", mcnemar(np.array([[bothright, YesNo], [NoYes, bothwrong]]), correction=False))
    # print("statsmodels exact false correction false", mcnemar(np.array([[bothright, YesNo], [NoYes, bothwrong]]), exact=False, correction=False))
    sig = mcnemar(np.array([[bothright, YesNo], [NoYes, bothwrong]]), exact=False, correction=False)
    return sig.pvalue

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

individs = {0 : 'src_para', 1 : 'src_orig', 2 : 'orig_para', 3 : "align_para"}

def logreg(names, outfile):
    results = []
    whole_corrects = {}
    for nam in tqdm.tqdm(names):
        embed = nam[0].split('_')
        embed_type = embed[0]
        composition = embed[-1].replace('david', 'joint').replace('para', 'n/a').replace('orig', 'n/a')

        X = metrics[nam].values
        y = metrics['annotation'].values
        groups = np.asarray([labels.get(x, labels['unk']) for x in metrics['label'].values])
        # gkf = GroupKFold(n_splits=2)
        total = len(set(groups))
        gkf = GroupKFold(n_splits=total)

        fold_accs = []
        fold_waccs = []
        fold_aps = []
        fold_waps = []
        fold_corrects = []
        


        whole_y = []
        whole_score = []

        for train, test in gkf.split(X, y, groups=groups):
            # print("Currently on {}".format(current))
            clf = LogisticRegression(random_state=0, solver='lbfgs', n_jobs=8, verbose=0) #, max_iter=50)
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
            # pdb.set_trace()
            weight = len(test_y) / len(y)
            if 1 in test_y and 0 in test_y:
                fold_aps.append(avp(test_y, is_para_prob))
                avg_prec = avp(test_y, is_para_prob)
                fold_aps.append(avg_prec)
                weight = test_y.shape[0] / y.shape[0]
                fold_waps.append(avg_prec * weight)
            y_hat = clf.predict(test_X)
            corrects = test_y == y_hat
            fold_corrects += list(corrects)
            acc = clf.score(test_X, test_y)
            
            fold_accs.append(acc)
            fold_waccs.append(acc * weight)
            # current += 1

        whole_ap = avp(whole_y, whole_score)
        results.append((embed_type,
                        composition,
                        str(np.mean(fold_accs)),
                        str(sum(fold_waccs)),
                        str(whole_ap),
                        str(np.mean(fold_aps)),
                        str(sum(fold_waps))))
        # print("fold_aps:", fold_apsn)

        whole_corrects['_'.join([embed_type, composition])] = fold_corrects

    for res in results:
        print('\t'.join(res))
        outfile.write('\t'.join(res) + '\n')
    
    return whole_corrects

outfile = open('acc_avp_and_map.tsv', 'w')
print('\t'.join(['embed', 'comp', 'acc', 'wacc', 'aveP', 'map', 'wmap']))
outfile.write('\t'.join(['embed', 'comp', 'acc', 'wacc', 'aveP', 'map', 'wmap']) + '\n')
print("all rels")
outfile.write("all rels\n")
all_corrs = logreg(names, outfile)
corrs = {}
corrs['all rels'] = all_corrs

for num in individs:
    name = individs[num]
    print("just {}".format(name))
    outfile.write("just {}\n".format(name))
    new_names = [[x[num]] for x in names]
    # pdb.set_trace()
    name_corrs = logreg(new_names, outfile)
    corrs[name] = name_corrs

# print('specials')
# print("just {} and {}".format(individs[2], individs[3]))
# outfile.write("just {} and {}".format(individs[2], individs[3]))
# new_names = [[x[2:]] for x in names]
# logreg(new_names, outfile)

to_nparray = []

for rel1 in corrs:
    for rel2 in corrs:
        for metric1 in corrs[rel1]:
            for metric2 in corrs[rel2]:
                if rel1 != rel2 or metric1 != metric2:
                    if corrs[rel1][metric1] != corrs[rel2][metric2]:
                        sig = signif(
                            corrs[rel1][metric1],
                            corrs[rel2][metric2]
                            )
                    else:
                        sig = 0.0
                    acc1 = sum(corrs[rel1][metric1]) / len(corrs[rel1][metric1])
                    acc2 = sum(corrs[rel2][metric2]) / len(corrs[rel2][metric2])
                    acc_diff = acc1 - acc2
                    to_nparray.append([
                        '_'.join([metric1, rel1]),
                        '_'.join([metric2, rel2]),
                        str(sig),
                        str(acc_diff)])

with open('pivot.tsv', 'w') as of:
    of.write('\t'.join(['from', 'to', 'sig', 'acc diff']) + '\n')
    for line in to_nparray:
        of.write('\t'.join(line) + '\n')

pdb.set_trace()
