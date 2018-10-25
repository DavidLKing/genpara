#!/usr/bin/env python3

import sys
import pdb
import numpy as np
import gensim
import allennlp
from allennlp.modules.elmo import Elmo, batch_to_ids

class Score:

    def __init__(self):
        pass

    def sim(self, a, b):
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        return cos_sim

    def mean_maker(self, sent, model):
        vecs = []
        for word in sent.split():
            if word in model:
                vecs.append(model[word])
        # print(sent)
        # pdb.set_trace()
        mean = np.mean(np.asarray(vecs), axis=0)
        return mean

    def david_metric(self, cos, dist):
        return (1 - ((cos + 1) / 2)) * dist

    def elmo_diffs(self, elmo, data):
        newdata = []
        print("generating elmo vectors")
        # srcs = elmo(batch_to_ids([x[2] for x in data]))['elmo_representations'][2]
        # paras = elmo(batch_to_ids([x[3] for x in data]))['elmo_representations'][2]
        # for line, src_vec, para_vec in zip(data, srcs, paras):
        for line in data:
            src_word = line[0]
            para_word = line[1]
            src = line[2]
            # tgt = line[4]
            para = line[3]
            # try:
            src_idx = src.split().index(src_word)
            para_idx = para.split().index(para_word)
            # except:
            #     pdb.set_trace()
            dist, cos, joint = self.elmo_word_diff(elmo, src, para, src_idx, para_idx)#, src_vec, para_vec)
            line = [str(dist), str(cos), str(joint)] + line
            # print(dist, cos, joint)
            print('\t'.join(line))
            newdata.append(line)
            # pdb.set_trace()
        return newdata

    def elmo_diffs_batch(self, elmo, data):
        newdata = []
        print("generating elmo vectors")
        srcs = elmo(batch_to_ids([x[2] for x in data]))['elmo_representations'][2]
        paras = elmo(batch_to_ids([x[3] for x in data]))['elmo_representations'][2]
        for line, src_vec, para_vec in zip(data, srcs, paras):
            src_word = line[0]
            para_word = line[1]
            src = line[2]
            # tgt = line[4]
            para = line[3]
            # try:
            src_idx = src.split().index(src_word)
            para_idx = para.split().index(para_word)
            # except:
            #     pdb.set_trace()
            dist, cos, joint = self.elmo_word_diff(elmo, src, para, src_idx, para_idx, src_vec, para_vec)
            line = [str(dist), str(cos), str(joint)] + line
            print('\t'.join(line))
            # print(dist, cos, joint)
            newdata.append(line)
            # pdb.set_trace()
        return newdata


    def elmo_word_diff(self, model, src, para, src_idx, para_idx):#, src_vec, para_vec):
        # src_vecs = model(batch_to_ids([src]))
        # elmo(batch_to_ids([src]))['elmo_representations'][2]
        src_word_vec = np.asarray(model(batch_to_ids([src]))['elmo_representations'][2][0][src_idx].detach())
        # para_vecs = model(batch_to_ids([para]))
        para_word_vec = np.asarray(model(batch_to_ids([para]))['elmo_representations'][2][0][para_idx].detach())
        # pdb.set_trace()
        dist = np.linalg.norm(src_word_vec - para_word_vec)
        cos = self.sim(src_word_vec, para_word_vec)
        joint = self.david_metric(cos, dist)
        # pdb.set_trace()
        return dist, cos, joint

    def score(self, line, word2vec, glove):
        # defunct
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

# w2v = gensim.models.KeyedVectors.load_word2vec_format(sys.argv[2], binary=True)
# glove = gensim.models.KeyedVectors.load_word2vec_format(sys.argv[3], binary=False)
# infile = open(sys.argv[1], 'r').readlines()
#
# for line in infile:
#     w2v_sim, glove_sim, w2v_dist, glove_dist, w2v_david, glove_david = score(line, w2v, glove)
#     print('\t'.join([str(w2v_sim), str(glove_sim), str(w2v_dist), str(glove_dist), str(w2v_david), str(glove_david)] + line.split('\t')))
