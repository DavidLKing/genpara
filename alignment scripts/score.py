#!/usr/bin/env python3

import sys
import pdb
import h5py
import numpy as np
import gensim
import pandas
import allennlp
from allennlp.modules.elmo import Elmo, batch_to_ids

from amad_batcher import MiniBatch

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

    def rebuild_dialogs(self, corrected):
        dialog = 0
        turn = 0
        num_src = {}
        for sent in [x.split('\t')[0] for x in corrected]:
            turn += 1
            if sent.startswith("#START"):
                dialog += 1
                turn = 0
            if sent not in num_src:
                num_src[sent] = []
            num_src[sent].append([dialog - 1, turn - 1])
        return num_src


    def elmo_word_diff(self, model, src, para, src_idx, para_idx):#, src_vec, para_vec):
        # src_vecs = model(batch_to_ids([src]))
        # elmo(batch_to_ids([src]))['elmo_representations'][2]
        src_word_vec = np.asarray([2][0][src_idx].detach())
        # src_word_vec = np.asarray(model(batch_to_ids([src]))['elmo_representations'][2][0][src_idx].detach())
        # para_vecs = model(batch_to_ids([para]))
        para_word_vec = np.asarray(model(batch_to_ids([para]))['elmo_representations'][2][0][para_idx].detach())
        # para_word_vec = np.asarray(model(batch_to_ids([para]))['elmo_representations'][2][0][para_idx].detach())
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

    def score(self, line, word2vec, glove, elmo_src, elmo_aligned, elmo_orig, elmo_para):
        # defunct
        line = line.lower().strip().split('\t')
        swappable = line[0]
        # swapped = line[1]
        para_word = line[1]
        src = line[2].split(' ')
        aligned = line[3].split(' ')
        orig = line[5].split(' ')
        para = line[4].split(' ')
        # src_word = line[0]
        src_idx = src.index(swappable)
        align_idx = aligned.index(para_word)
        orig_idx = orig.index(swappable)
        para_idx = para.index(para_word)
        # try:
        # elmo_src_vec = np.asarray(elmo_src[src_idx].detach())
        # elmo_align_vec = np.asarray(elmo_aligned[align_idx].detach())
        # elmo_orig_vec = np.asarray(elmo_orig[orig_idx].detach())
        # elmo_para_vec = np.asarray(elmo_para[para_idx].detach())
        elmo_src_vec = elmo_src[src_idx]
        elmo_align_vec = elmo_aligned[align_idx]
        elmo_orig_vec = elmo_orig[orig_idx]
        elmo_para_vec = elmo_para[para_idx]
        # except:
        #     pdb.set_trace()
        # ELMO SRC -> PARA
        elmo_src_para_sim, elmo_src_para_dist, elmo_src_para_david = self.score_list(elmo_src_vec, elmo_para_vec)
        # ELMO SRC -> ORIG
        elmo_src_orig_sim, elmo_src_orig_dist, elmo_src_orig_david = self.score_list(elmo_src_vec, elmo_orig_vec)
        # ELMO ORIG -> PARA
        elmo_orig_para_sim, elmo_orig_para_dist, elmo_orig_para_david = self.score_list(elmo_orig_vec, elmo_para_vec)
        # ELMO ALIGN -> PARA
        elmo_align_para_sim, elmo_align_para_dist, elmo_align_para_david = self.score_list(elmo_align_vec, elmo_para_vec)
        # GLOVE
        glove_src = self.mean_maker(src, glove)
        glove_aligned = self.mean_maker(aligned, glove)
        glove_orig = self.mean_maker(orig, glove)
        glove_para = self.mean_maker(para, glove)
        # W2V
        w2v_src = self.mean_maker(src, word2vec)
        w2v_align = self.mean_maker(aligned, word2vec)
        w2v_orig = self.mean_maker(orig, word2vec)
        w2v_para = self.mean_maker(para, word2vec)
        # TODO is this enought checks or should I check EVERY vector?
        if not np.isnan(glove_para[0]) and not np.isnan(glove_src[0]) and \
                not np.isnan(w2v_para[0]) and not np.isnan(w2v_src[0]):
            # pdb.set_trace()
            # GLOVE SRC -> PARA
            glove_src_para_sim = self.sim(glove_src, glove_para)
            glove_src_para_dist = np.linalg.norm(glove_src - glove_para)
            glove_src_para_david = self.david_metric(glove_src_para_sim, glove_src_para_dist)
            # GLOVE SRC -> ORIG
            glove_src_orig_sim = self.sim(glove_src, glove_orig)
            glove_src_orig_dist = np.linalg.norm(glove_src - glove_orig)
            glove_src_orig_david = self.david_metric(glove_src_orig_sim, glove_src_orig_dist)
            # GLOVE ORIG -> PARA
            glove_orig_para_sim = self.sim(glove_orig, glove_para)
            glove_orig_para_dist = np.linalg.norm(glove_orig - glove_para)
            glove_orig_para_david = self.david_metric(glove_orig_para_sim, glove_orig_para_dist)
            # GLOVE ALIGN -> PARA
            glove_align_para_sim = self.sim(glove_aligned, glove_para)
            glove_align_para_dist = np.linalg.norm(glove_aligned - glove_para)
            glove_align_para_david = self.david_metric(glove_align_para_sim, glove_align_para_dist)
            # W2V SRC -> PARA
            w2v_src_para_sim = self.sim(w2v_src, w2v_para)
            w2v_src_para_dist = np.linalg.norm(w2v_src - w2v_para)
            w2v_src_para_david = self.david_metric(w2v_src_para_sim, w2v_src_para_dist)
            # W2V SRC -> ORIG
            w2v_src_orig_sim = self.sim(w2v_src, w2v_orig)
            w2v_src_orig_dist = np.linalg.norm(w2v_src - w2v_orig)
            w2v_src_orig_david = self.david_metric(w2v_src_orig_sim, w2v_src_orig_dist)
            # W2V ORIG -> PARA
            w2v_orig_para_sim = self.sim(w2v_orig, w2v_para)
            w2v_orig_para_dist = np.linalg.norm(w2v_orig - w2v_para)
            w2v_orig_para_david = self.david_metric(w2v_orig_para_sim, w2v_orig_para_dist)
            # W2V ALIGN -> PARA
            w2v_align_para_sim = self.sim(w2v_align, w2v_para)
            w2v_align_para_dist = np.linalg.norm(w2v_align - w2v_para)
            w2v_align_para_david = self.david_metric(w2v_align_para_sim, w2v_align_para_dist)
            # pdb.set_trace()
            return (glove_src_para_sim,
                glove_src_para_dist,
                glove_src_para_david,
                glove_src_orig_sim,
                glove_src_orig_dist,
                glove_src_orig_david,
                glove_orig_para_sim,
                glove_orig_para_dist,
                glove_orig_para_david,
                glove_align_para_sim,
                glove_align_para_dist,
                glove_align_para_david,
                w2v_src_para_sim,
                w2v_src_para_dist,
                w2v_src_para_david,
                w2v_src_orig_sim,
                w2v_src_orig_dist,
                w2v_src_orig_david,
                w2v_orig_para_sim,
                w2v_orig_para_dist,
                w2v_orig_para_david,
                w2v_align_para_sim,
                w2v_align_para_dist,
                w2v_align_para_david,
                elmo_src_para_sim, 
                elmo_src_para_dist, 
                elmo_src_para_david,
                elmo_src_orig_sim, 
                elmo_src_orig_dist, 
                elmo_src_orig_david,
                elmo_orig_para_sim, 
                elmo_orig_para_dist, 
                elmo_orig_para_david,
                elmo_align_para_sim, 
                elmo_align_para_dist, 
                elmo_align_para_david)



s = Score()

print("""
    hacky arguments:
    1 = w2v binary file
    2 = glove text file in w2v format
    3 = tsv to score
    4 = original corrected.tsv (2016 VP data)
        """)

corrected = open(sys.argv[4], 'r').readlines()
dialog_turn_nums = s.rebuild_dialogs(corrected)

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
options_file = "../data/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "../data/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
elmo = Elmo(options_file, weight_file, 3, dropout=0)
m = MiniBatch(options_file,
              weight_file,
              3,
              device=0)
# elmo_src = h5py.File(sys.argv[3], 'r')
# elmo_src = h5py.File('elmo_singular_swap.src.h5py', 'r')
# elmo_tgt = h5py.File(sys.argv[4], 'r')
# elmo_para = h5py.File('elmo_singular_swap.tgt.h5py', 'r')
# testing minibatching
# test_sents = [x.strip().split(' ') for x in open('elmo_singular_swap.src', 'r').readlines()]
# test_extract = m.extract(test_sents, 3, 32)

swap_txt = open(sys.argv[3], 'r').readlines()
print("extracting ELMo representations...")
swap_csv = pandas.read_csv(sys.argv[3] ,delimiter='\t')
# pdb.set_trace()
srcs = [x.split() for x in swap_csv['src'].tolist()]
aligns = [x.split() for x in swap_csv['align'].tolist()]
origs = [x.split() for x in swap_csv['orig'].tolist()]
paras = [x.split() for x in swap_csv['para'].tolist()]
# Non-minibatched
# elmo_src = elmo(batch_to_ids(srcs))['elmo_representations'][2]
# elmo_align = elmo(batch_to_ids(aligns))['elmo_representations'][2]
# elmo_orig = elmo(batch_to_ids(origs))['elmo_representations'][2]
# elmo_para = elmo(batch_to_ids(paras))['elmo_representations'][2]
# mini-batched
# TODO make 3rd arg, batch size, an option

batch_size = 128

print("Extracting ELMo rep for srcs")
elmo_src = m.extract(srcs, 2, batch_size)
print("Extracting ELMo rep for aligns")
elmo_align = m.extract(aligns, 2, batch_size)
print("Extracting ELMo rep for origs")
elmo_orig = m.extract(origs, 2, batch_size)
print("Extracting ELMo rep for paras")
elmo_para = m.extract(paras, 2, batch_size)


# TODO make this an option or get logging to a different file
outfile = open('scored.tsv', 'w')

header = swap_txt[0].strip().split('\t')

header = ['dialog',
            'turn',
            'glove_src_para_sim',
            'glove_src_para_dist',
            'glove_src_para_david',
            'glove_src_orig_sim',
            'glove_src_orig_dist',
            'glove_src_orig_david',
            'glove_orig_para_sim',
            'glove_orig_para_dist',
            'glove_orig_para_david',
            'glove_align_para_sim',
            'glove_align_para_dist',
            'glove_align_para_david',
            'w2v_src_para_sim',
            'w2v_src_para_dist',
            'w2v_src_para_david',
            'w2v_src_orig_sim',
            'w2v_src_orig_dist',
            'w2v_src_orig_david',
            'w2v_orig_para_sim',
            'w2v_orig_para_dist',
            'w2v_orig_para_david',
            'w2v_align_para_sim',
            'w2v_align_para_dist',
            'w2v_align_para_david',
            'elmo_src_para_sim', 
            'elmo_src_para_dist', 
            'elmo_src_para_david',
            'elmo_src_orig_sim', 
            'elmo_src_orig_dist', 
            'elmo_src_orig_david',
            'elmo_orig_para_sim', 
            'elmo_orig_para_dist', 
            'elmo_orig_para_david',
            'elmo_align_para_sim', 
            'elmo_align_para_dist', 
            'elmo_align_para_david'] + header
# print('\t'.join(header))
outfile.write('\t'.join(header) + '\n')


line_nmr = 0
missing = 0
total = 0

lost = []

# TODO add sanity check to make sure lengths are all correct
# TODO figure out how to get Pandas to output something I can iterate through
for line in swap_txt[1:]:
    total += 1
    line = line.lower()
    # w2v_sim, glove_sim, elmo_sim, w2v_dist, glove_dist, elmo_dist, w2v_david, glove_david, elmo_david = s.score(line, w2v, glove, elmo_src[str(line_nmr)], elmo_para[str(line_nmr)])
    sims = s.score(line, w2v, glove,
               elmo_src[line_nmr],
               elmo_align[line_nmr],
               elmo_orig[line_nmr],
               elmo_para[line_nmr])
    # print('sims', sims)
    # print('line', line)
    # print('\t'.join(list([str(x) for x in sims]) + line.strip().split('\t')))
    original = line.split('\t')[5]
    if original in dialog_turn_nums:
        for nums in dialog_turn_nums[original]:
            dial_num = nums[0]
            turn_num = nums[1]
            outline = '\t'.join([str(dial_num), str(turn_num)] + list([str(x) for x in sims]) + line.strip().split('\t')) + '\n'
            if 'nan nan' in outline:
                pdb.set_trace()
            else:
                outfile.write(outline)
    else:
        lost.append('\t'.join(list([str(x) for x in sims]) + line.strip().split('\t')) + '\n')
        missing += 1
    line_nmr += 1

print("missing", missing, "of", total)







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
VE SRC -> ORIG
            # GLOVE ORIG -> PARA
                        # GLOVE ALIGN -> PARA
                        
'''
