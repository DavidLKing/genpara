#!/usr/bin/env python3

import sys
import pdb
import numpy as np
import gensim
import allennlp

def sim(a, b):
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return cos_sim

def mean_maker(sent, model):
    vecs = []
    for word in sent.split():
        if word in model:
            vecs.append(model[word])
    # print(sent)
    # pdb.set_trace()
    mean = np.mean(np.asarray(vecs), axis=0)
    return mean

def david_metric(cos, dist):
    return (1 - ((cos + 1) / 2)) * dist

def score(line, word2vec, glove):
    para = line.strip().split('\t')[0]
    sent = line.strip().split('\t')[1]
    glove_para = mean_maker(para, glove)
    glove_sent = mean_maker(sent, glove)
    w2v_para = mean_maker(para, word2vec)
    w2v_sent = mean_maker(sent, word2vec)
    # pdb.set_trace()
    if not np.isnan(glove_para[0]) and not np.isnan(glove_sent[0]) and \
       not np.isnan(w2v_para[0]) and not np.isnan(w2v_sent[0]):
        glove_sim = sim(glove_sent, glove_para)
        w2v_sim = sim(w2v_sent, w2v_para)
        glove_dist = np.linalg.norm((glove_sent, glove_para))
        w2v_dist = np.linalg.norm((w2v_sent, w2v_para))
        glove_david = david_metric(glove_sim, glove_dist)
        w2v_david = david_metric(w2v_sim, w2v_dist)
        return w2v_sim, glove_sim, w2v_dist, glove_dist, w2v_david, glove_david

w2v = gensim.models.KeyedVectors.load_word2vec_format(sys.argv[2], binary=True)
# w2v = []
glove = gensim.models.KeyedVectors.load_word2vec_format(sys.argv[3], binary=False)
infile = open(sys.argv[1], 'r').readlines()

for line in infile:
    # print(line.split('\t')[0])
    w2v_sim, glove_sim, w2v_dist, glove_dist, w2v_david, glove_david = score(line, w2v, glove)
    print('\t'.join([str(w2v_sim), str(glove_sim), str(w2v_dist), str(glove_dist), str(w2v_david), str(glove_david)] + line.split('\t')))
