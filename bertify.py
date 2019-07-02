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

  def __init__(self, corrs, annos, cutoff, binary=True, dev_p=0.1):
    self.binary = binary
    self.dev_p = dev_p
    self.cutoff = cutoff
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
    self.smoothz = SmoothingFunction().method4
    
    self.main(corrs, annos)

  def meteor_score(self, sent1, sent2):
    # DEPRECIATED, DOESN'T WORK
    # Write to Meteor: SCORE ||| sent1 ||| sent2
    inputs = "SCORE ||| {} ||| {}\n".format(sent1, sent2)
    self.meteor.stdin.write(inputs.encode())
    # Read stats from Meteor
    stats = self.meteor.stdout.readline()  # still ends with \n
    # Write to Meteor: EVAL ||| stats
    self.meteor.stdin.write("EVAL ||| {}".format(stats))
    # Read score from Meteor
    score = self.meteor.stdout.readline()  # still ends with \n
    # Write score to Lamtram
    # sys.stdout.write(score)
    # Must flush after every line to avoid process communication deadlock
    sys.stdout.flush()
    return str(score)

  # def rank_job(self, src, tgt, label1, label2, annotation):
  def rank_job(self, pos):
    # self.prog_bar.update(1)
    try:
      score = bleu(pos[0], pos[1], smoothing_function=self.smoothz)
      # score = bleu(src, tgt, smoothing_function=self.smoothz)
      # return score, [src, tgt, label1, label2, annotation]
      return score, pos
    except:
      pass


  def rank_and_extrace(self, possibles, reverse=False):
    total = len(possibles)
    threads = mp.cpu_count() - 1

    # testing
    # possibles = possibles[0:5000]

    # self.prog_bar = tqdm.tqdm(total=len(possibles))
    p = mp.Pool(threads)
    # scored = p.starmap(self.rank_job, possibles)
    # scored = p.imap(self.rank_job, possibles)
    # scored2 = tqdm.tqdm(p.imap(self.rank_job, possibles), total=len(possibles))
    # for pos in tqdm.tqdm(possibles):
    #   self.rank_job(pos)

    print("Scoring")

    scored = []
    for pos in tqdm.tqdm(p.imap_unordered(self.rank_job, possibles), total=total):
      if pos != None:
        scored.append(pos)
    p.close()
    p.join()
    # pdb.set_trace()

    print("Sorting")

    sorted(scored, key=lambda x: x[0])
    if reverse:
      scored.reverse()
    return [x[1] for x in scored]


  def gen_pos(self, labels):
    # Enumerate all possible positive examples:
    positives = []
    for label in labels:
      for sent1 in labels[label]:
        for sent2 in labels[label]:
          if sent1 != sent2:
            positives.append((sent1, sent2, label, label, 1))
            positives.append((sent2, sent1, label, label, 1))
    positives = self.rank_and_extrace(positives, reverse=False)
    return positives
  
  def gen_neg(self, labels):
    # Enumerate all possible negative examples:
    negatives = []
    for label1 in labels:
      for label2 in labels:
        if label1 != label2:
          for sent1 in labels[label1]:
            for sent2 in labels[label2]:
              if sent1 != sent2:
                negatives.append((sent1, sent2, label1, label2, 0))
                negatives.append((sent2, sent1, label2, label1, 0))
    negatives = self.rank_and_extrace(negatives, reverse=True)
    return negatives


  def gen_corrected(self, filename):
    labels = {}
    with open(filename, 'r') as fl:
      duplicates = 0
      total = 0
      for line in fl:
        line = line.strip().split('\t')
        if len(line) > 1:
          total += 1
          sent = line[0]
          label = line[1]
          if label not in labels:
            labels[label] = []
          if sent in labels[label]:
            duplicates += 1
          labels[label].append(sent)
    print("Total duplicates {} of {} total".format(duplicates, total))
    print("Getting true paraphrases")
    positives = self.gen_pos(labels)
    limit = int(self.cutoff * len(positives))
    posses = positives[0:limit]
    print("Getting false paraphrases")
    negs = self.gen_neg(labels)
    chosen_negs = negs[0:limit]
    # pdb.set_trace()
    # currently sampling without replacement randomly with uniform dist. 
    # TODO make more sophisticated later
    # chosen_negs = random.sample(negs, k=len(posses))
    assert(len(chosen_negs) == len(posses))
    # not very efficient. job gets killed. use random instead
    # chosen_negs = np.random.choice(negs, len(posses), replace=False)
    # SHUFFLE o'clock!
    random.shuffle(posses)
    random.shuffle(chosen_negs)
    dev_num = int(len(posses) * self.dev_p)
    train_pos = posses[dev_num:]
    train_neg = chosen_negs[dev_num:]
    dev_pos = posses[0:dev_num]
    dev_neg = chosen_negs[0:dev_num]
    train = train_pos + train_neg
    dev = dev_pos + dev_neg
    random.shuffle(train)
    random.shuffle(dev)
    return train, dev

  def gen_annos(self, filename):
    annos = []
    with open(filename, 'r') as of:
      for line in of:
        line = line.strip().split('\t')
        orig = line[5]
        para = line[4]
        anno = line[-1]
        label = line[6]
        annos.append((orig, para, label, label, anno))
    # remove original header
    annos.pop(0)
    return annos

  def clean(self, train, dev, test):
    print("Sizes before cleaning:\n train {} \n dev {} \n test {}".format(len(train), len(dev), len(test)))
    testitems = set()
    for t in test:
      testitems.add(t[0])
      testitems.add(t[1])
    newtrain = []
    newdev = []
    for item in train:
      if item[0].lower() not in testitems and item[1].lower() not in testitems:
        newtrain.append(item)
    for item in dev:
      if item[0].lower() not in testitems and item[1].lower() not in testitems:
        newdev.append(item)
    print("Sizes after cleaning:\n new train {} \n new dev {} \n test {}".format(len(newtrain), len(newdev), len(test)))
    return newtrain, newdev, test




  def main(self, corrs, annos):
    header = "\t".join(['index', 'sentence1', 'sentence2', 'label']) + '\n'
    train, dev = self.gen_corrected(corrs)
    test = self.gen_annos(annos)
    train, dev, test = self.clean(train, dev, test)
    # pdb.set_trace()
    index = 0
    with open('train.tsv', 'w') as of:
      of.write(header)
      for tupe in train:
        of.write('\t'.join([str(index), tupe[0], tupe[1], str(tupe[-1])]) + '\n')
        index += 1
    index = 0
    with open('dev.tsv', 'w') as of:
      of.write(header)
      for tupe in dev:
        of.write('\t'.join([str(index), tupe[0], tupe[1], str(tupe[-1])]) + '\n')
        index += 1
    index = 0
    with open('test.tsv', 'w') as of:
      of.write(header)
      for tupe in test:
        of.write('\t'.join([str(index), tupe[0], tupe[1], str(tupe[-1])]) + '\n')
        index += 1


Bertify(sys.argv[1], sys.argv[2], float(sys.argv[3]))


