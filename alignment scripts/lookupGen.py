#!/usr/bin/env python3

import sys
import pickle as pkl
import pdb

lookup = {}

print("Building lookup table")
for line in open('..//data/UD_English-EWT/ewt.txt', 'r'):
    if not line.startswith("#"):
        line = line.split()
        if len(line) > 3:
            lemma = line[2]
            wordform = line[1]
            pos = line[4]
            if lemma not in lookup:
                lookup[lemma] = {}
            if pos not in lookup[lemma]:
                lookup[lemma][pos] = wordform.lower()

print("Pickling!")
outfile = open('lookup.pkl', 'wb')
pkl.dump(lookup, outfile)