#!/usr/bin/env python3

import glob
import pdb
import sys
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

dev = glob.glob(sys.argv[1] + '/*')
sys1 = glob.glob(sys.argv[2] + '/*')
sys2 = glob.glob(sys.argv[3] + '/*')

corrects = []

for f in dev:
    ifile = open(f + '/dev.tsv', 'r')
    lines = ifile.readlines()
    for line in lines:
        line = line.strip().split('\t')
        corr = line[-1]
        if corr != 'label':
            corrects.append(corr)
    ifile.close()

preds1 = []
preds2 = []

for f1, f2 in zip(sys1, sys2):
    ofile1 = open(f1 + '/scores.txt', 'r')
    ofile2 = open(f2 + '/scores.txt', 'r')
    for line1, line2 in zip(ofile1, ofile2):
        # try:
        line1 = line1.strip()
        line1 = line1.split('\t')
        # assert(len(line1) == 2)
        line2 = line2.strip()
        line2 = line2.split('\t')
        # assert(len(line2) == 2)
        if len(line1) > 1 and len(line2) > 1:
            pred1 = line1[1]
            score1 = line1[0]
            preds1.append(pred1)

            pred2 = line2[1]
            score2 = line2[0]
            preds2.append(pred2)
            # except:
                # continue

    ofile1.close()
    ofile2.close()

def correct(system, gold):
    correct = 0
    total = 0
    for l1, l2 in zip(system, gold):
        total += 1
        if l1 == l2:
            correct += 1
    # print("Acc:", correct / total)
    return correct / total

# pdb.set_trace()
print(sys.argv[2] + " acc:", correct(preds1, corrects))
print(sys.argv[3] + " acc:", correct(preds2, corrects))


bothright = 0
YesNo = 0
NoYes = 0
bothwrong = 0
for c1, c2, gold in zip(preds1, preds2, corrects):
    if c1 == gold and c2 == gold:
        bothright += 1
    elif c1 == gold and c2 != gold:
        YesNo += 1
    elif c1 != gold and c2 == gold:
        NoYes += 1
    elif c1 != gold and c2 != gold:
        bothwrong += 1
    else:
        print("Error")
        pdb.set_trace()


print("statsmodels", mcnemar(np.array([[bothright, YesNo], [NoYes, bothwrong]])))
print("statsmodels exact false", mcnemar(np.array([[bothright, YesNo], [NoYes, bothwrong]]), exact=False))
print("statsmodels correction false", mcnemar(np.array([[bothright, YesNo], [NoYes, bothwrong]]), correction=False))
print("statsmodels exact false correction false", mcnemar(np.array([[bothright, YesNo], [NoYes, bothwrong]]), exact=False, correction=False))
