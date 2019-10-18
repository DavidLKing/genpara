# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 16:50:26 2019

@author: 13982
"""

"""
Produced file format:
    [0]: aligned indices in left_ind - right_ind
    [1]: aligned words
    [2]: POS (left-right)
    [3]: DEP (left-right)
    [4]: BERT left vector
    [5]: BERT right vector
    [6]: gold label
    [7]: the longer of sentence span
    ...(UPDATE THIS TABLE ONCE SOME CHANGES HAPPEN!!!!)
"""

import torch
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM
import spacy
from scipy.spatial import distance
import sys, csv, re, pdb
import numpy as np
from spacy.symbols import ORTH, LEMMA, POS, TAG
from collections import defaultdict

from bert import BertBatch
from classifier import NeuralClassifier
from processing import DataProcessing

class TrainingData():
    
    def __init__(self, training_file):
        self.training_file = training_file
        self.BERT_INFO = []
        self.punctuation = ["'", ".", ",", "!", "?", "-"]
        
    def train_bert(self):
        self.processing = DataProcessing()
        self.nlp = spacy.load('en_core_web_sm')
        """Cutstomize the sentence segmentation at last"""
        """★Refer to 'sentence segmentation' and 'add_pipe' section on SPACY'"""
        self.nlp.add_pipe(self.processing.set_custom_boundaries, before='parser')
        """Hard-code some special cases where words are tokenized differently from the gold data"""
        """★However, sometimes spacy is actually correct, this is a makeshift method, we should actually improve our data"""
        special_case = [{ORTH: 'cannot'}]
        self.nlp.tokenizer.add_special_case('cannot', special_case)
        special_case = [{ORTH: 'youve'}]
        self.nlp.tokenizer.add_special_case('youve', special_case)
        """https://spacy.io/usage/linguistic-features#native-tokenizers"""
        """以上为customize的全教程，合理运用！！！！"""
        prefix_re = re.compile(r'''''')
        suffix_re = re.compile(r'''''')
        infix_re = re.compile(r'''[~]''')
        self.nlp.tokenizer.infix_finditer = infix_re.finditer
        self.nlp.tokenizer.suffix_search = suffix_re.search
        self.nlp.tokenizer.prefix_search = prefix_re.search
        
        trials = []
        sentSet = set()
        sentences = []
        goldScores = []
        with open(self.training_file) as tsv:
            for line in csv.reader(tsv, dialect="excel-tab"):
                if len(line) > 0:
                    combo = " ".join(line)
                    """Omit repetitive combo"""
                    if combo not in sentSet:
                        align_pairs = line[2].split()
                        
                        
                        sent1 = line[0]
                        sent2 = line[1]
                        
                        """Below are preprocessing for spacy"""
                        sentence1 = list(self.nlp(sent1).sents)[0]
                        sentence2 = list(self.nlp(sent2).sents)[0]
                        
                        """★★★★暂时改回原来的"""
                        s1 = sent1.split()
                        s2 = sent2.split()
                        
                        
                        
                        """Add in the shifted gold label alignments before actually splitting sentences"""
                        """DEBUGGING: in order to distinguish whether a punctuation is split before or after our split operation"""
                        gold1 = []
                        gold2 = []
                        for i in range(0, len(sentence1)):
                            gold1.append([])
                        for i in range(0, len(sentence2)):
                            gold2.append([])
                            
                            """For all pairs in gold alignments, add the aligned index to the corresponding index of the list"""
                        for align in align_pairs:
                            pair = align.split('-')
                            left_ind = int(pair[0])
                            right_ind = int(pair[1])
                            gold1[left_ind].append(right_ind)
                            
                        """★结果证明：标点符号不会被直接align，相反没有发现align的！！！！"""
                        sentences.append(s1)
                        sentences.append(s2)
                        
                        """Represent the shifted alignments in the same format as gold label"""
                        shifted_gold = ""
                        for i, j in enumerate(gold1):
                            if len(gold1[i]) > 0:
                                for ii, k in enumerate(j):
                                    if i != 0 or ii >= 1:
                                        shifted_gold += " "
                                    shifted_gold += str(i) + "-" + str(k)
                        goldScores.append(set(shifted_gold.split()))
                        
                        sentSet.add(combo)
                        trials.append([])
                        
        batch_size = 10
        # pdb.set_trace()
        i = 0
        batch_loc = 0
        # while batch_loc < len(self.sentences):
        #     pdb.set_trace()
        # batches = self.sentences[batch_loc : (batch_loc + batch_size)]

        if torch.cuda.is_available():
            b = BertBatch(device=0)
        else:
            b = BertBatch(device=-1)

        embeddings = b.extract(self.sentences, batch_size, self.layer)

        j = 0
        while (j+1) < len(embeddings):
            
            gold = goldScores[int(i/2)]
            
            sent1 = list(sentences[i])
            sent2 = list(sentences[i+1])
            
            """★★★★注意检查：left和right到底是不是正确代表了左右句？？？？"""
            
            left = embeddings[j]
            right = embeddings[j+1]
            
            left_num_words = len(sent1)
            right_num_words = len(sent2)
            
#                print("sent1: ")
#                print(sent1)
#                print("sent2： ")
#                print(sent2)
#                
#                print(left)
#                print(right)
#                
#                print(left.shape)
#                print(right.shape)
            
            assert (left_num_words, right_num_words) == (len(sent1), len(sent2))
            
            lNumWords = len(sent1)
            rNumWords = len(sent2)
            
            alignments = []
            alignsRead = []
            
            avg_alignments = []
            avg_alignsRead = []
            
            """TRAINING: Record the larger of the sentence length"""
            sent_len = max([len(sent1), len(sent2)])
            
            lTOr = {}
            rTOl = {}
            aligned_set = set()
#                print("Reach before single alignment")
            
            for row in range(left_num_words):
                for col in range(right_num_words):
                    left_vector = left[row]
                    right_vector = right[col]
                    
                    """这个条件式其实可以去掉了"""
                    if row < len(sent1) and col < len(sent2):
                        """The row and col in cell matches the two sentences"""
                        alignString = str(row) + "-" + str(col)
                        alignWords = str(sent1[row]) + " | " + str(sent2[col])
#                            print("Involved index: ")
#                            print(alignString)
#                            print("Involved words: ")
#                            print(alignWords)
                        
#                            print("Left vector")
#                            print(left_vector)
#                            print("Right vector")
#                            print(right_vector)
                        
                        aligned_set.add(alignString)
                        gold_label = self.processing.is_gold(alignString, gold)
                        
                        """只要gold label有，就加入考虑"""
                        if gold_label:
                            lTOr[row] = col
                            rTOl[col] = row
#                                print("Block middle reached")
                            alignments.append(alignString)
                            alignsRead.append(alignWords)
                            
                            avg_alignments.append(alignString)
                            avg_alignsRead.append(alignWords)
                            
                        """TRAINING: Get the token tuples (index, POS, DEP, text)"""
                        tuple1 = self.processing.get_POS_and_DEP(' '.join(sent1), self.nlp, [row])
                        tuple2 = self.processing.get_POS_and_DEP(' '.join(sent2), self.nlp, [col])
                        pos = str(self.processing.get_POS(tuple1, [row])) + '-' + str(self.processing.get_POS(tuple2, [col]))
                        dep = str(self.processing.get_DEP(tuple1, [row])) + '-' + str(self.processing.get_DEP(tuple2, [col]))
                        self.BERT_INFO.append((alignString, alignWords, pos, dep, left_vector, right_vector, gold_label, sent_len))
                        
#                        print("--------------------------Single Separator---------------------------")
                        
                        
                    """暂时不考虑multi-alignment！！！！"""
                        
                        
            nulls = self.processing.hasNull(alignments, lNumWords, rNumWords)
            
            allAligns = self.processing.alignsToRead(alignments, sent1, sent2)
            alignments = allAligns[0]
            alignsRead = allAligns[1]
            
#                print(alignments)
#                print(alignsRead)
            
            
            
            
            
            
            
            
            
            
#                """以下预留给multi-alignment部分"""
#                
#                """★追加一段与已经aligned的进行multi align的block！！！！"""
#                l_branch = self.processing.depAnalyze(' '.join(sent1), alignments, nulls[0], self.nlp, which='left')
#                r_branch = self.processing.depAnalyze(' '.join(sent2), alignments, nulls[1], self.nlp, which='right')
#                    
##                print(l_branch)
##                print(r_branch)
#                
#                for lb in l_branch:
##                    print("ENTERING DEP ANALYSIS!!!!")
#                    l_core = lb[0]
#                    aligned = lTOr[l_core]
##                    print(aligned)
#                    for rb in r_branch:
#                        r_core = rb[0]
#                        if r_core == aligned:
#                            aligned = rb.copy()
#                    l_ind = lb.copy()
#                    
#                    """Dealing with bugs concerning single aligned case"""
#                    if type(aligned) != list:
#                        aligned = [aligned]
#                    r_ind = aligned.copy()
#                    
#                    """TESTING: 去掉全部aligned的indices！！！！"""
#                    if len(l_ind) != 2 or len(r_ind) != 1:
#                        for l in l_ind:
#                            if l in lTOr:
#                                l_ind.remove(l)
#                        for r in r_ind:
#                            if r in rTOl:
#                                r_ind.remove(r)
#                                
#                    print(aligned)
#                    
#                    """DEBUGGING: solve the issue of empty alignments"""
#                    if len(l_ind) == 0 or len(r_ind) == 0:
#                        continue
#                    
#                    l_embedding = self.processing.avgEmbedding(l_ind, r_ind, left, right, sent1, sent2, which='left')
#                    r_embedding = self.processing.avgEmbedding(l_ind, r_ind, left, right, sent1, sent2, which='right')
#                                
#                    written_str = str([str(sent1[l]) for l in l_ind]) + '|' + str([str(sent2[r]) for r in r_ind])
#                    
##                    print(str(l_ind) + '----' + str(r_ind))
##                    print(str(written_str))
#                    
#                    written = str(l_ind) + '----' + str(r_ind)
#                    tuple1 = self.processing.get_POS_and_DEP(' '.join(sent1), self.nlp, l_ind)
#                    tuple2 = self.processing.get_POS_and_DEP(' '.join(sent2), self.nlp, r_ind)
#                    pos = str(self.processing.get_POS(tuple1, l_ind)) + '-' + str(self.processing.get_POS(tuple2, r_ind))
#                    dep = str(self.processing.get_DEP(tuple1, l_ind)) + '-' + str(self.processing.get_DEP(tuple2, r_ind))
#                    gold_label = self.processing.is_gold(written, gold)
#                    self.BERT_INFO.append((written, written_str, pos, dep, l_embedding, r_embedding, gold_label, sent_len))
#                    
#                
#                
#                """以上预留给multi-alignment部分，等single好了再说（到时候直接从原文档复制粘贴）"""
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            """allAligns的[0]是word-word版本，[1]是index-index版本"""
            avg_allAligns = self.processing.alignsToRead(avg_alignments, sent1, sent2)
            avg_alignments = avg_allAligns[0]
            avg_alignsRead = avg_allAligns[1]
            
            """每次除以2再取整就能保证trails中同样的index包含同一对句子的两个不同的alignment pair"""
            trials[int(i/2)].append(allAligns)
            trials[int(i/2)].append(avg_allAligns)
#                print(trials[int(i/2)])
            
            i += 2
            j += 2
            
#        batch_loc += batch_size

    def get_info(self):
        return self.BERT_INFO

    def get_nlp(self):
        return self.nlp


