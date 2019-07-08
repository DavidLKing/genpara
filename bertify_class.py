#!/usr/bin/env python3

import sys
import random
import tqdm
import os
import multiprocessing as mp
import pdb
import subprocess
import nltk
from nltk.translate import bleu
from nltk.translate.bleu_score import SmoothingFunction
import numpy as np

class Bertify:

  def __init__(self, corr2016, corr2017, binary=True, dev_p=0.1):
    self.binary = binary
    self.dev_p = dev_p
    # self.cutoff = 0.2
    # assert(self.annos == True or self.corrs == True)
    # assert(self.annos != self.corrs)
    # Borrowed
    # meteor_jar = "~/bin/meteor-1.5/meteor-1.5.jar"
    # meteor_args = "-lower -l en -t tune"
    # self.meteor = subprocess.Popen(
      # "java -Xmx2G -jar {} - - -stdio {}".format(
      #   meteor_jar, 
      #   meteor_args
      # ),
      # shell=True,
      # stdin=subprocess.PIPE,
      # stdout=subprocess.PIPE
    # )
    # self.smoothz = SmoothingFunction().method4
    self.main(corr2016, corr2017)

  def gen_corrected(self, filename, labels):
    # turn = 0
    dialogs = []
    with open(filename, 'r') as fl:
      for line in fl:
        line = line.strip().split('\t')
        if len(line) > 1:
          sent = line[0]
          label = line[1]
          labels.add(label)
          dialogs.append([sent, label])
    return dialogs, labels

  def conf_set_to_dicts(self, label_set):
    idx = 0
    label2idx = {}
    idx2label = {}
    for label in label_set:
      if label not in label2idx:
        label2idx[label] = idx
        idx2label[idx] = label
        idx += 1
    return label2idx, idx2label


  def main(self, corr2016, corr2017):
    header = "\t".join(['sentence', 'label']) + '\n'
    labels = set()
    c2016, labels = self.gen_corrected(corr2016, labels)
    c2017, labels = self.gen_corrected(corr2017, labels)
    label2idx, idx2label = self.conf_set_to_dicts(labels)

    train_dev = [[x[0], str(label2idx[x[1]])] for x in c2016]
    test = [[x[0], str(label2idx[x[1]])] for x in c2017]

    random.shuffle(train_dev)
    random.shuffle(test)

    limit = int(len(train_dev) * self.dev_p)
    train = train_dev[limit:]
    dev = train_dev[0:limit]

    with open('train.tsv', 'w') as of:
      of.write(header)
      for tupe in train:
        of.write('\t'.join(tupe) + '\n')
    with open('dev.tsv', 'w') as of:
      of.write(header)
      for tupe in dev:
        of.write('\t'.join(tupe) + '\n')
    with open('test.tsv', 'w') as of:
      of.write(header)
      for tupe in test:
        of.write('\t'.join(tupe) + '\n')


Bertify(sys.argv[1], sys.argv[2])


