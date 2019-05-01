#!/usr/bin/env python3

import sys
import pdb
import h5py
import numpy as np
import gensim
import pandas
import kenlm
import allennlp
from allennlp.modules.elmo import Elmo, batch_to_ids

from amad_batcher import MiniBatch

class lang_mod:

    def __init__(self):
        pass

    def load_lm(self, lmfile):
        print("Loading language model", lmfile)
        # print(lmfile[-5:])
        if lmfile[-4:] == 'arpa':
            models = arpa.loadf(lmfile)
            self.lm = models[0]
            self.score = lambda x: self.lm.log_p(x)
        elif lmfile[-3:] == 'bin':
            self.lm = kenlm.LanguageModel(lmfile)
            self.score = lambda x: self.lm.score(x)
        else:
            sys.exit("I can only read kenlm .bin or .arpa file")


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
        mean = np.mean(np.asarray(vecs), axis=0)
        return mean

    def phrasal_mean_maker(self, sent, model):
        # also works for single words
        vecs = []
        for word in sent:
            if word in range(len(model)):
                vecs.append(model[word])
        # if vecs == []:
        #     print('vecs = []')
        #     pdb.set_trace()
        # print(sent)
        # pdb.set_trace()
        mean = np.mean(np.asarray(vecs), axis=0)
        if type(mean) != np.ndarray:
            if np.isnan(mean):
                print("We got a nan!")
                print("\tvecs", vecs)
                print("\tsent", sent)
                pdb.set_trace()
        # should we check every value for nans?
        return mean

    def david_metric(self, cos, dist):
        return (1 - ((cos + 1) / 2)) * dist

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

    def score_list(self, src_vec, para_vec):
        cos_sim = self.sim(src_vec, para_vec)
        dist = np.linalg.norm(src_vec - para_vec)
        joint = self.david_metric(cos_sim, dist)
        return cos_sim, dist, joint

    def return_idxs(self, sent, word_list):
        idxs = []
        # for just getting pattern words
        for word in word_list:
                try:
                    idxs.append(sent.index(word))
                    # idxs.append(sent.index(word) + 1)
                except:
                    print('word', word)
                    print('sent', sent)
                    pdb.set_trace()
        return idxs

    def return_other_idxs(self, sent, word_list):
        idxs = []
        # for just getting variable words
        for word in sent:
            if word not in word_list:
                try:
                    idxs.append(sent.index(word))
                    # idxs.append(sent.index(word) + 1)
                except:
                    print('word', word)
                    print('sent', sent)
                    pdb.set_trace()
        return idxs

    def list_of_words(self, swapper):
        try:
            swapper = [x for x in eval(swapper) if "$" not in x]
        except:
            swapper = [swapper]
        return swapper

    def score(self, line, word2vec, glove, ng_model, composition_type, elmo_src, elmo_aligned, elmo_orig, elmo_para):
        # defunct
        line = line.lower().strip().split('\t')
        swappable = self.list_of_words(line[0])
        # print('swappable line', line[0])
        # should acount for sarah's and my stuff
        # swapped = line[1]
        para_word = self.list_of_words(line[1])
        # print('para_word line', line[1])
        src = line[2].split(' ')
        aligned = line[3].split(' ')
        # again, for sarah's output
        try:
            orig = eval(line[5])
        except:
            orig = line[5].split(' ')
        para = line[4].split(' ')
        # src_word = line[0]
        # print('swappable', swappable)
        # print('para_word', para_word)
        # print('src', src)
        # print('orig', orig)
        # print('aligned', aligned)
        # print('para', para)
        # print("doing src")
        # return indexes from word in pattern
        if composition_type == 'average':
            # averaging... why not?
            src_idxes = [x for x in range(len(src))]
        elif composition_type == 'variable':
            src_idxes = self.return_other_idxs(src, swappable)
        elif composition_type == 'lexsub':
            src_idxes = self.return_idxs(src, swappable)


        # print('src_idxes', src_idxes)
        # pdb.set_trace()
        # src_idx = src.index(swappable)
        # print("doing align")
        # return indexes from word in pattern
        if composition_type == 'average':
            # averaging... why not?
            align_idxes = [x for x in range(len(aligned))]
        elif composition_type == 'variable':
            align_idxes = self.return_other_idxs(aligned, para_word)
        elif composition_type == 'lexsub':
            align_idxes = self.return_idxs(aligned, para_word)

        # print('align_idxes', align_idxes)
        # pdb.set_trace()
        # align_idx = aligned.index(para_word)
        # print("doing orig")
        # return indexes from word not in pattern
        if composition_type == 'average':
            # averaging... why not?
            orig_idxes = [x for x in range(len(orig))]
        elif composition_type == 'variable':
            orig_idxes = self.return_other_idxs(orig, swappable)
        elif composition_type == 'lexsub':
            orig_idxes = self.return_idxs(orig, swappable)

        # print('orig_idxes', orig_idxes)
        # pdb.set_trace()
        # orig_idx = orig.index(swappable)
        # print("doing para")
        # return indexes from word not in pattern
        if composition_type == 'average':
            # averaging... why not?
            para_idxes = [x for x in range(len(para))]
        elif composition_type == 'variable':
            para_idxes = self.return_other_idxs(para, para_word)
        elif composition_type == 'lexsub':
            para_idxes = self.return_idxs(para, para_word)


        # print('para_idxes', para_idxes)
        # pdb.set_trace()
        # para_idx = para.index(para_word)
        # try:
        # elmo_src_vec = np.asarray(elmo_src[src_idx].detach())
        # elmo_align_vec = np.asarray(elmo_aligned[align_idx].detach())
        # elmo_orig_vec = np.asarray(elmo_orig[orig_idx].detach())
        # elmo_para_vec = np.asarray(elmo_para[para_idx].detach())
        # print("doing src")
        # pdb.set_trace()
        elmo_src_vec = self.phrasal_mean_maker(src_idxes, elmo_src)
        # pdb.set_trace()
        # elmo_src_vec = elmo_src[src_idx]
        # print("doing align")
        elmo_align_vec = self.phrasal_mean_maker(align_idxes, elmo_aligned)
        # elmo_align_vec = elmo_aligned[align_idx]
        # print("doing orig")
        elmo_orig_vec = self.phrasal_mean_maker(orig_idxes, elmo_orig)
        # elmo_orig_vec = elmo_orig[orig_idx]
        # print("doing para")
        elmo_para_vec = self.phrasal_mean_maker(para_idxes, elmo_para)
        # elmo_para_vec = elmo_para[para_idx]
        # except:
        # pdb.set_trace()
        # ELMO SRC -> PARA
        elmo_src_para_sim, elmo_src_para_dist, elmo_src_para_david = self.score_list(elmo_src_vec, elmo_para_vec)
        # ELMO SRC -> ORIG
        elmo_src_orig_sim, elmo_src_orig_dist, elmo_src_orig_david = self.score_list(elmo_src_vec, elmo_orig_vec)
        # ELMO ORIG -> PARA
        elmo_orig_para_sim, elmo_orig_para_dist, elmo_orig_para_david = self.score_list(elmo_orig_vec, elmo_para_vec)
        # ELMO ALIGN -> PARA
        elmo_align_para_sim, elmo_align_para_dist, elmo_align_para_david = self.score_list(elmo_align_vec, elmo_para_vec)
        elmo_sims = elmo_src_para_sim + elmo_src_orig_sim + elmo_orig_para_sim + elmo_align_para_sim
        elmo_dist = elmo_src_para_dist + elmo_src_orig_dist + elmo_orig_para_dist + elmo_align_para_dist
        elmo_david = elmo_src_para_david + elmo_src_orig_david + elmo_orig_para_david + elmo_align_para_david
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
        # ng_src = ng_model.score(' '.join(src))
        # ng_orig = ng_model.score(' '.join(aligned))
        # ng_align = ng_model.score(' '.join(orig))
        # ng_para = ng_model.score(' '.join(para))
        ng_src = ng_model.lm.perplexity(' '.join(src))
        ng_orig = ng_model.lm.perplexity(' '.join(aligned))
        ng_align = ng_model.lm.perplexity(' '.join(orig))
        ng_para = ng_model.lm.perplexity(' '.join(para))
        ng_src_para = ng_src - ng_para
        ng_src_orig = ng_src - ng_orig
        ng_orig_para = ng_orig - ng_para
        ng_align_para = ng_align - ng_para
        ng_sum = ng_src_orig + ng_src_para + ng_orig_para + ng_align_para
        # pdb.set_trace()
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
                elmo_align_para_david,
                elmo_sims,
                elmo_dist,
                elmo_david,
                ng_src_para,
                ng_src_orig,
                ng_orig_para,
                ng_align_para,
                ng_sum)



s = Score()


print("""
    hacky arguments:
    1 = w2v binary file
    2 = glove text file in w2v format
    3 = tsv to score
    4 = original corrected.tsv (2016 VP data)
    5 = Gigaword 5-gram model
    6 = lexsub/variable/average
    7 = specials/noSpecials
        """)


# temp test

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
# options_file = "../data/elmo_2x4096_512_2048cnn_2xhighway_options.json"
# weight_file = "../data/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
options_file = "../data/bigelmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
weight_file = "../data/bigelmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"
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

# KENLM
print('loading kenlm gigaword 5gram model')
lm = lang_mod()
lm.load_lm(sys.argv[5])

composition_type = sys.argv[6]
assert(composition_type in ['lexsub', 'average', 'variable'])

specials = sys.argv[7]
assert(specials in ['specials', 'noSpecials'])

swap_txt = open(sys.argv[3], 'r').readlines()
print("extracting ELMo representations...")
swap_csv = pandas.read_csv(sys.argv[3] ,delimiter='\t')
# pdb.set_trace()
srcs = [x.split() for x in swap_csv['src'].tolist()]
aligns = [x.split() for x in swap_csv['align'].tolist()]
origs = [x.split() for x in swap_csv['orig'].tolist()]
# for sarah's output
if type(origs[0]) == list:
    origs = [' '.join(x).replace("['", "").replace("']", "").replace("', '", " ") for x in origs]
paras = [x.split() for x in swap_csv['para'].tolist()]
# Non-minibatched
# elmo_src = elmo(batch_to_ids(srcs))['elmo_representations'][2]
# elmo_align = elmo(batch_to_ids(aligns))['elmo_representations'][2]
# elmo_orig = elmo(batch_to_ids(origs))['elmo_representations'][2]
# elmo_para = elmo(batch_to_ids(paras))['elmo_representations'][2]
# mini-batched
# TODO make 3rd arg, batch size, an option

batch_size = 1

# "warm up"
# TODO during actual run with real data (i.e. no sanity data) only run once
print('warming up')
# pdb.set_trace()
m.extract(srcs, 2, batch_size, specials)
m.extract(aligns, 2, batch_size, specials)
m.extract(origs, 2, batch_size, specials)
m.extract(paras, 2, batch_size, specials)

print("Extracting ELMo rep for srcs")
elmo_src = m.extract(srcs, 2, batch_size, specials)
print("Extracting ELMo rep for aligns")
elmo_align = m.extract(aligns, 2, batch_size, specials)
print("Extracting ELMo rep for origs")
elmo_orig = m.extract(origs, 2, batch_size, specials)
print("Extracting ELMo rep for paras")
elmo_para = m.extract(paras, 2, batch_size, specials)


# TODO make this an option or get logging to a different file
outfile = open('scored.tsv', 'w')

header = swap_txt[0].strip().split('\t')

# header = [
header = ['glove_src_para_sim',
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
            'elmo_align_para_david',
            'elmo_sims',
            'elmo_dist',
            'elmo_david',
            'ng_src_para',
            'ng_src_orig',
            'ng_orig_para',
            'ng_align_para',
            'ng_sum'] + header
# print('\t'.join(header))
outfile.write('\t'.join(header) + '\n')


line_nmr = 0
missing = 0
total = 0

lost = []

# TODO add sanity check to make sure lengths are all correct
# TODO figure out how to get Pandas to output something I can iterate through
for line in swap_txt[1:]:
    print("number", total)
    total += 1
    line = line.lower()
    # w2v_sim, glove_sim, elmo_sim, w2v_dist, glove_dist, elmo_dist, w2v_david, glove_david, elmo_david = s.score(line, w2v, glove, elmo_src[str(line_nmr)], elmo_para[str(line_nmr)])
    # score(self, line, word2vec, glove, ng_model, elmo_src, elmo_aligned, elmo_orig, elmo_para)
    sims = s.score(line, w2v, glove, lm, composition_type,
                   elmo_src[line_nmr],
                   elmo_align[line_nmr],
                   elmo_orig[line_nmr],
                   elmo_para[line_nmr])
    # pdb.set_trace()
    # print('sims', sims)
    # print('line', line)
    # print('\t'.join(list([str(x) for x in sims]) + line.strip().split('\t')))
    split_line = line.strip().split('\t')
    outfile.write('\t'.join(list([str(x) for x in sims]) + line.strip().split('\t')) + '\n')
    line_nmr += 1
