#!/usr/bin/env python3

import sys
import pdb
import h5py
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
        for word in sent:
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

    def score_list(self, src_vec, para_vec):
        cos_sim = self.sim(src_vec, para_vec)
        dist = np.linalg.norm(src_vec - para_vec)
        joint = self.david_metric(cos_sim, dist)
        return cos_sim, dist, joint

    def score(self, line, word2vec, glove, elmo_src, elmo_para):
        # defunct
        line = line.strip().split('\t')
        para = line[3].split(' ')
        sent = line[2].split(' ')
        # src_word = line[0]
        para_word = line[1]
        src_idx = sent.index(para_word)
        para_idx = para.index(para_word)
        elmo_src_vec = elmo_src[src_idx]
        elmo_para_vec = elmo_para[para_idx]
        elmo_sim, elmo_dist, elmo_david = self.score_list(elmo_src_vec, elmo_para_vec)
        glove_para =self.mean_maker(para, glove)
        glove_sent = self.mean_maker(sent, glove)
        w2v_para = self.mean_maker(para, word2vec)
        w2v_sent = self.mean_maker(sent, word2vec)
        # pdb.set_trace()
        if not np.isnan(glove_para[0]) and not np.isnan(glove_sent[0]) and \
           not np.isnan(w2v_para[0]) and not np.isnan(w2v_sent[0]):
            # pdb.set_trace()
            glove_sim = self.sim(glove_sent, glove_para)
            w2v_sim = self.sim(w2v_sent, w2v_para)
            glove_dist = np.linalg.norm(glove_sent - glove_para)
            w2v_dist = np.linalg.norm(w2v_sent - w2v_para)
            glove_david = self.david_metric(glove_sim, glove_dist)
            w2v_david = self.david_metric(w2v_sim, w2v_dist)
            return w2v_sim, glove_sim, elmo_sim, w2v_dist, glove_dist, elmo_dist, w2v_david, glove_david, elmo_david

# w2v = gensim.models.KeyedVectors.load_word2vec_format(sys.argv[2], binary=True)
# glove = gensim.models.KeyedVectors.load_word2vec_format(sys.argv[3], binary=False)
# infile = open(sys.argv[1], 'r').readlines()
#
# for line in infile:
#     w2v_sim, glove_sim, w2v_dist, glove_dist, w2v_david, glove_david = score(line, w2v, glove)
#     print('\t'.join([str(w2v_sim), str(glove_sim), str(w2v_dist), str(glove_dist), str(w2v_david), str(glove_david)] + line.split('\t')))

s = Score()

### W2V ###
print("loading W2V vectors")
# currently commented out for processing time
w2v = gensim.models.KeyedVectors.load_word2vec_format(sys.argv[1], binary=True)
# w2v = gensim.models.KeyedVectors.load_word2vec_format('../data/vectors.300.bin', binary=True)
# w2v = ''

### GloVe ###
print("loading GloVe vectors")
# currently commented out for processing time
glove = gensim.models.KeyedVectors.load_word2vec_format(sys.argv[2], binary=False)
# glove = gensim.models.KeyedVectors.load_word2vec_format('../data/glove.6B.300d.txt.word2vec', binary=False)
# glove = ''

### ELMO ###
print("loading ELMo...")
elmo_src = h5py.File(sys.argv[3], 'r')
# elmo_src = h5py.File('elmo_singular_swap.src.h5py', 'r')
elmo_tgt = h5py.File(sys.argv[4], 'r')
# elmo_para = h5py.File('elmo_singular_swap.tgt.h5py', 'r')

swap_csv = open(sys.argv[5], 'r').readlines()
# swap_csv = open('elmo_singular_swap.tsv', 'r').readlines()

line_nmr = 0
# TODO add sanity check to make sure lengths are all correct
for line in swap_csv:
    # w2v_sim, glove_sim, elmo_sim, w2v_dist, glove_dist, elmo_dist, w2v_david, glove_david, elmo_david = s.score(line, w2v, glove, elmo_src[str(line_nmr)], elmo_para[str(line_nmr)])
    sims = s.score(line, w2v, glove, elmo_src[str(line_nmr)], elmo_tgt[str(line_nmr)])
    print('\t'.join(list([str(x) for x in sims]) + line.strip().split('\t')))
    line_nmr += 1









### OLD STUFF ###
# to do: download these
# options_file = "../data/elmo_2x4096_512_2048cnn_2xhighway_options.json"
# weight_file = "../data/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
# options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
# weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

# elmo = Elmo(options_file, weight_file, 3, dropout=0)
# messy prototype
# sent = "This is a test"
# single_sent_elmo = lambda sent: elmo(batch_to_ids([sent]))
'''
(Pdb) len(single_sent_elmo("This is a test"))
2
(Pdb) single_sent_elmo("This is a test").keys()
dict_keys(['elmo_representations', 'mask'])
(Pdb) single_sent_elmo("This is a test")['mask']
tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
(Pdb) single_sent_elmo("This is a test")['elmo_representations'].shape
*** AttributeError: 'list' object has no attribute 'shape'
(Pdb) len(single_sent_elmo("This is a test")['elmo_representations'])
2
(Pdb) single_sent_elmo("This is a test")['elmo_representations'][0]
tensor([[[-4.1971e-01, -4.4025e-01, -1.0587e+00,  ...,  7.2652e-02,
          -7.4392e-01, -4.4630e-01],
         [-3.3414e-01, -7.8741e-01, -4.8827e-01,  ...,  7.9663e-02,
           3.8118e-01,  1.9112e-02],
         [-1.1728e-01, -2.9004e-01, -5.1561e-01,  ..., -1.3266e-01,
          -9.3138e-02, -2.9839e-01],
         ...,
         [ 2.0640e-01,  3.3418e-01, -4.8526e-02,  ...,  4.1011e-02,
           7.4620e-01,  9.6554e-02],
         [ 3.9084e-01,  6.2279e-01, -1.3639e-01,  ...,  4.2772e-01,
           6.9154e-01, -2.4551e-01],
         [ 8.0042e-02, -9.9265e-02, -4.2147e-01,  ..., -3.7287e-01,
           1.8058e-01,  2.3906e-01]]], grad_fn=<DropoutBackward>)
(Pdb) single_sent_elmo("This is a test")['elmo_representations'][0].shape
torch.Size([1, 14, 1024])
(Pdb) single_sent_elmo("This is a test")['elmo_representations'][1].shape
torch.Size([1, 14, 1024])
(Pdb) single_sent_elmo("This is a test")['elmo_representations'][2].shape
*** IndexError: list index out of range
(Pdb) 
'''