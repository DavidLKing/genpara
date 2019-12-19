#!/usr/bin/env python3

import pdb
import sys
import tqdm
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn
from statsmodels.stats.contingency_tables import mcnemar
import os
from pandas import Series, DataFrame

file1 = pd.read_csv(sys.argv[1], delimiter=',')
file2 = pd.read_csv(sys.argv[2], delimiter=',')

corrs1 = file1['correct']
corrs2 = file2['correct']

golds = open(sys.argv[3], 'r').readlines()
training_labels = open('rares.txt', 'r').readlines()
rares = [x.strip().split() for x in training_labels]
rares = [x[1] for x in rares if int(x[0]) < 21]

bothright = 0
YesNo = 0
NoYes = 0
bothwrong = 0

acc1 = 0
acc2 = 0
total = 0

rare_bothright = 0
rare_YesNo = 0
rare_NoYes = 0
rare_bothwrong = 0

rare_acc1 = 0
rare_acc2 = 0
rare_total = 0

statistic = lambda x,y: (x - y)**2 / (x + y)

for c1, c2, true in zip(corrs1, corrs2, golds):
    gold_label = true.split('\t')[0]
    if gold_label not in ['359', '360']:
        if c1 == True and c2 == True:
            bothright += 1
            acc1 += 1
            acc2 += 1
            if gold_label in rares:
                rare_bothright += 1
                rare_acc1 += 1
                rare_acc2 += 1
        elif c1 == True and c2 == False:
            YesNo += 1
            acc1 += 1
            if gold_label in rares:
                rare_YesNo += 1
                rare_acc1 += 1
        elif c1 == False and c2 == True:
            NoYes += 1
            acc2 += 1
            if gold_label in rares:
                rare_NoYes += 1
                rare_acc2 += 1
        elif c1 == False and c2 == False:
            bothwrong += 1
            if gold_label in rares:
                rare_bothwrong += 1
        else:
            print("Error")
            pdb.set_trace()
        total += 1
        if gold_label in rares:
            rare_total += 1

print("sys1 acc:", acc1 / total)
print("sys2 acc:", acc2 / total)

print("manual:", statistic(YesNo, NoYes))
print("statsmodels", mcnemar(np.array([[bothright, YesNo], [NoYes, bothwrong]])))
print("statsmodels exact false", mcnemar(np.array([[bothright, YesNo], [NoYes, bothwrong]]), exact=False))
print("statsmodels correction false", mcnemar(np.array([[bothright, YesNo], [NoYes, bothwrong]]), correction=False))
print("statsmodels exact false correction false", mcnemar(np.array([[bothright, YesNo], [NoYes, bothwrong]]), exact=False, correction=False))

print('\nrare labels!!!')
print('\nthere are {} total labels'.format(total))
print("sys1 acc:", rare_acc1 / rare_total)
print("sys2 acc:", rare_acc2 / rare_total)

print("manual:", statistic(rare_YesNo, rare_NoYes))
print("statsmodels", mcnemar(np.array([[rare_bothright, rare_YesNo], [rare_NoYes, rare_bothwrong]])))
print("statsmodels exact false", mcnemar(np.array([[rare_bothright, rare_YesNo], [rare_NoYes, rare_bothwrong]]), exact=False))
print("statsmodels correction false", mcnemar(np.array([[rare_bothright, rare_YesNo], [rare_NoYes, rare_bothwrong]]), correction=False))
print("statsmodels exact false correction false", mcnemar(np.array([[rare_bothright, rare_YesNo], [rare_NoYes, rare_bothwrong]]), exact=False, correction=False))
