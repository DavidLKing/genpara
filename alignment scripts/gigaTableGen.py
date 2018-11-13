#!/usr/bin/env python3

import sys
import pickle as pkl
import pdb
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

lookup = {}

print("Building lookup table")
# for line in open('../data/gigaword/test.txt', 'r'):
for line in open('../data/gigaword/gigaword-4.txt', 'r'):
    text_block = []
    line = nltk.word_tokenize(line.lower())
    line = nltk.pos_tag(line)
    for word in line:
        wordform = word[0]
        lemma = lemmatizer.lemmatize(wordform)
        pos = word[1]
        if lemma not in lookup:
            lookup[lemma] = {}
        if pos not in lookup[lemma]:
            lookup[lemma][pos] = wordform.lower()

print("Pickling!")
outfile = open('lookup.pkl', 'wb')
pkl.dump(lookup, outfile)
