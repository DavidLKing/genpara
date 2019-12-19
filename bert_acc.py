#!/usr/bin/env python3

import glob
import pdb
import sys

inputfiles = glob.glob(sys.argv[1] + '/*')
outputfiles = glob.glob(sys.argv[2] + '/*')

line_nums = []

for f in inputfiles:
    ifile = open(f + '/dev.tsv', 'r')
    lines = ifile.readlines()
    line_nums.append(len(lines))
    ifile.close()

accs = []

for f in outputfiles:
    ofile = open(f + '/eval_results.txt', 'r')
    for line in ofile:
        line = line.strip()
        if line.startswith("acc"):
            line = line.split(' ')
            assert(len(line) == 3)
            accs.append(float(line[2]))
    ofile.close()

assert(len(line_nums) == len(accs))

print("Raw acc:", sum(accs) / len(accs))

waccs = []
total = sum(line_nums)
for acc, num in zip(accs, line_nums):
    waccs.append(acc * (num / total))
print("Weighted acc:", sum(waccs))

