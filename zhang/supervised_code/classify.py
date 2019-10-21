# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 17:11:01 2019

@author: 13982
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
"""DataProcessing module contains every function to process data"""
from processing import DataProcessing

"""????SEEMS TO WORK: Feed all vector values to the input matrix for certain example_index"""
def feedVectorToInput(input_matrix, vector, example_index, dimension, which='left'):    
    for i in range(0, len(vector)):
        if which == 'left':
            input_matrix[example_index][i] = vector[i]
        elif which == 'right':
            input_matrix[example_index][dimension+i] = vector[i]

class PredictingData():
    
    def __init__(self, predicting_file, neural_model, nlp, classifier_dimension):
        self.predicting_file = predicting_file
        self.model = neural_model
        self.punctuation = ["'", ".", ",", "!", "?", "-"]
        self.nlp = nlp
        self.trials = []
        self.goldScores = []
        self.sentences = []
        self.dimension = classifier_dimension
        self.processing = DataProcessing()
        
    def predict(self):
        sentSet = set()
        with open(self.predicting_file) as tsv:
            for line in csv.reader(tsv, dialect="excel-tab"):
                if len(line) > 0:
                    combo = " ".join(line)
                    """跳过重复的"""
                    if combo not in sentSet:
                        align_pairs = line[2].split()
                        
                        sent1 = line[0]
                        sent2 = line[1]
                        
                        """Below are preprocessing for spacy"""
                        sentence1 = list(self.nlp(sent1).sents)[0]
                        sentence2 = list(self.nlp(sent2).sents)[0]
                        
                        """★Maybe resolved... Testing"""
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
                        self.sentences.append(s1)
                        self.sentences.append(s2)
                        
                        """Represent the shifted alignments in the same format as gold label"""
                        shifted_gold = ""
                        for i, j in enumerate(gold1):
                            if len(gold1[i]) > 0:
                                for ii, k in enumerate(j):
                                    if i != 0 or ii >= 1:
                                        shifted_gold += " "
                                    shifted_gold += str(i) + "-" + str(k)
                        self.goldScores.append(set(shifted_gold.split()))
                        
                        sentSet.add(combo)
                        self.trials.append([])
                        
        batch_size = 10
        i = 0
#        batch_loc = 0
        # while batch_loc < len(self.sentences):
        #     pdb.set_trace()
        # batches = self.sentences[batch_loc : (batch_loc + batch_size)]

        if torch.cuda.is_available():
            b = BertBatch(device=0)
        else:
            b = BertBatch(device=-1)

        embeddings = b.extract(self.sentences, batch_size)
        
        j = 0
        while (j+1) < len(embeddings):
            
            gold = self.goldScores[int(i/2)]
            
            sent1 = list(self.sentences[i])
            sent2 = list(self.sentences[i+1])
            
            left = embeddings[j]
            right = embeddings[j+1]
            
            left_num_words = len(sent1)
            right_num_words = len(sent2)
            
            print("sent1: ")
            print(sent1)
            print("sent2： ")
            print(sent2)
            
            """左右句分别的num of words！！！！"""
            lNumWords = len(sent1)
            rNumWords = len(sent2)
            
            alignments = []
            alignsRead = []
            
            avg_alignments = []
            avg_alignsRead = []
            
            """Record the larger of the sentence length"""
            sent_len = max([len(sent1), len(sent2)])
            
            lTOr = {}
            rTOl = {}
            aligned_set = set()
            print("Reach before single alignment")
            
            
            
            for row in range(left_num_words):
                """best_instance stores (row, column 1, score of row-0)"""
                """initialized to -1 so the first col will always rewrite the number, DEBUG"""
                best_instance = (row, 0, -1)
                for col in range(right_num_words):
                    left_vector = left[row]
                    right_vector = right[col]
                    """以下条件式先去掉吧 还没想到更好的替代"""
                    written = str(row) + '-' + str(col)
                    sent_distance = self.processing.getIndexDistanceNormalized(written, sent_len)
                    
                    """Extra Feature: Any feature beyond two embeddings"""
                    num_extra_features = 0
                    self.input_size = (2 * self.dimension) + num_extra_features
                    
                    input_value = np.zeros((1, self.input_size))
                    feedVectorToInput(input_value, left_vector, 0, self.dimension, which='left')
                    feedVectorToInput(input_value, right_vector, 0, self.dimension, which='right')
                    pred_value = self.model.predict(input_value)
                    
                    """Update the best score"""
                    if pred_value > best_instance[2]:
                        best_instance = (row, col, pred_value)
                    
                    alignString = str(row) + "-" + str(col)
                    alignWords = str(sent1[row]) + " | " + str(sent2[col])
                    
                    print("Involved index: ")
                    print(alignString)
                    print("Involved words: ")
                    print(alignWords)
                    print("Currently in single alignment")
                    print("Neural Prediction is [single]: ")
                    print(pred_value)
                    print("-------------Single Alignment Separator----------------")
                
                """Below are the 'best' row-col combination for each row"""
                best_row = best_instance[0]
                best_col = best_instance[1]
                best_combo = (row, col)
                best_score = best_instance[2]
                if best_score > 0.5 and best_combo not in aligned_set and sent1[best_row] not in self.punctuation and sent2[best_col] not in self.punctuation:
                    """The row and col in cell matches the two sentences"""
                    print("Oh Yeah! Singly Aligned!")
                    aligned_set.add(best_combo)
                    
                    lTOr[row] = col
                    rTOl[col] = row
                    
                    aligned_index = str(best_row) + "-" + str(best_col)
                    aligned_string = str(sent1[best_row]) + "|" + str(sent2[best_col])
                    
                    alignments.append(aligned_index)
                    alignsRead.append(aligned_string)
                    
                    avg_alignments.append(aligned_index)
                    avg_alignsRead.append(aligned_string)
                    
                    
            nulls = self.processing.hasNull(alignments, lNumWords, rNumWords)
            
            allAligns = self.processing.alignsToRead(alignments, sent1, sent2)
            alignments = allAligns[0]
            alignsRead = allAligns[1]
            
            print(alignments)
            print(alignsRead)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
#                """以下为multi部分，如果出现bug就在这里检查"""
#                
#                l_branch = self.processing.depAnalyze(' '.join(sent1), alignments, nulls[0], self.nlp, which='left')
#                r_branch = self.processing.depAnalyze(' '.join(sent2), alignments, nulls[1], self.nlp, which='right')
#                
#                print(l_branch)
#                print(r_branch)
#                
#                for lb in l_branch:
#                    print("ENTERING DEP ANALYSIS!!!!")
#                    l_core = lb[0]
#                    aligned = lTOr[l_core]
#                    print(aligned)
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
#                    
#                    """"""
#                    written = str(l_ind) + '----' + str(r_ind)
#                    sent_distance = self.processing.getIndexDistanceNormalized(written, sent_len)
#                    """Extra Feature: Any feature beyond two embeddings"""
#                    num_extra_features = 0
#                    self.input_size = (2 * self.dimension) + num_extra_features
#                    
#                    input_value = np.zeros((1, self.input_size))
#                    feedVectorToInput(input_value, l_embedding, 0, self.dimension, which='left')
#                    feedVectorToInput(input_value, r_embedding, 0, self.dimension, which='right')
#                    pred_value = self.model.predict(input_value)
#                    """"""
#                    
#                    print("Neural Prediction: ")
#                    print(pred_value)
#                    
#                    if pred_value > 0.5:
#                        for l in l_ind:
#                            for r in r_ind:
#                                """Avoid repetitive alignments"""
#                                align = str(l) + '-' + str(r)
#                                if align not in aligned_set and sent1[l] not in self.punctuation and sent2[r] not in self.punctuation:
#                                    aligned_set.add(align)
#                                    avg_alignments.append(str(l) + '-' + str(r))
#                                    avg_alignsRead.append(str(sent1[l]) + '|' + str(sent2[r]))
#                        
#                        written_str = str([str(sent1[l]) for l in l_ind]) + '|' + str([str(sent2[r]) for r in r_ind])
#                        
#                        print("DEP AVG ALIGNED!!")
#                        print(str(l_ind) + '----' + str(r_ind))
#                        print(str(written_str))
#                
#                
#                
#                """以上为multi部分，如果出现bug就在这里检查"""
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            """allAligns的[0]是word-word版本，[1]是index-index版本"""
            avg_allAligns = self.processing.alignsToRead(avg_alignments, sent1, sent2)
            avg_alignments = avg_allAligns[0]
            avg_alignsRead = avg_allAligns[1]
            
            """每次除以2再取整就能保证trails中同样的index包含同一对句子的两个不同的alignment pair"""
            self.trials[int(i/2)].append(allAligns)
            self.trials[int(i/2)].append(avg_allAligns)
            print(self.trials[int(i/2)])
            
            i += 2
            j += 2
            
#        batch_loc += batch_size
            
    def evaluate(self):
        TP_single = 0
        TP_multi = 0
        prec_single = 0
        prec_multi = 0
        rec_both = 0
        """Record the number of wins for single and multi alignments"""
        single_win = 0
        multi_win = 0
        tie = 0
        """开始比较每一对左右句的两次alignments的结果！！！！"""
        for i, pair in enumerate(self.trials):
            """Possible Values?"""
            posVals = []
            for best in pair:
                relativePos = 0
                alignments = best[0]
                alignsRead = best[1]
                """i刚好对应每一对左右句的index！！！！"""
                gold = self.goldScores[i]
                #pdb.set_trace()
                """对于两队alignments都是：一旦对上了gold，就加一个count！！！！"""
                for pair in alignments:
                    if pair in gold:
                        relativePos += 1
                """zero-division原来也可以用try except来解决！！！！"""
                """然后算出来当前alignments的precision，recall和F1！！！！"""
                try:
                    prec = relativePos/float(len(alignments))
                except:
                    prec = 0
                try:
                    rec = relativePos/float(len(gold))
                except:
                    rec = 0
                try:
                    relativeF1 = 2*((prec*rec)/(prec + rec))
                except:
                    relativeF1 = 0
                """将当前trial的数据存进trialTuple"""
                trialTuple = (relativePos, alignments, alignsRead, relativeF1, prec, rec)
                """然后posVals把每一个trialTuple存进去"""
                posVals.append(trialTuple)
            """要用n=i*2是因为每个i都存了 同一对左右句的两个trials！！！！ 所以用n才能表达每一对句子"""
            n = int((i*2))
            """但是这些句子不是已经hyphenated了吗？？？？没关系吗？？？？"""
            sent1 = self.sentences[n]
            sent2 = self.sentences[n+1]
            
            """最后把两个句子print出来！！！！单纯展示用"""
            sent1 = " ".join(sent1)
            sent2 = " ".join(sent2)
            print(sent1 + " || " + sent2)
            """posVals[0]是single word的，第二个posVals[1]才是multiword！！！！"""
            """[x][1]是index-index alignment， [x][2]是word-word alignment！！！！"""
            print("Single Word:")
            print(posVals[0][2])
            print(posVals[0][1])
            print("Potentially Multi-Word:")
            print(posVals[1][2])
            print(posVals[1][1])
            print("Gold: " + str(gold))
            
            TP_single += posVals[0][0]
            alignments_single = posVals[0][1]
            TP_multi += posVals[1][0]
            alignments_multi = posVals[1][1]
            
            """[x][3]是两个trial分别的F1-score，接下来比较谁的F1更高！！！！并将之变成真正的结果数据！！！！【完全懂了】"""
            if posVals[0][3] > posVals[1][3]:
                print("Trial 1 was better")
#                truePos += posVals[0][0]
#                alignments = posVals[0][1]
                single_win += 1
            elif posVals[0][3] < posVals[1][3]:
                print("Trial 2 was better")
#                truePos += posVals[1][0]
#                alignments = posVals[1][1]
                multi_win += 1
            else:
                print("Both Equal!!")
#                truePos += posVals[1][0]
#                alignments = posVals[1][1]
                tie += 1
            """proposed和actual算法还是一样"""
            prec_single += len(alignments_single)
            prec_multi += len(alignments_multi)
            rec_both += len(gold)
            #pdb.set_trace()
            print()
        
        precision_single = TP_single/float(prec_single)
        precision_multi = TP_multi/float(prec_multi)
        recall_single = TP_single/float(rec_both)
        recall_multi = TP_multi/float(rec_both)
        f1_single = 2*((precision_single*recall_single)/(precision_single + recall_single))
        f1_multi = 2*((precision_multi*recall_multi)/(precision_multi + recall_multi))
        print("Precision for single: " + str(precision_single))
        print("Recall for single: " + str(recall_single))
        print("F1 for single: " + str(f1_single))
        print("-------------------------------------------")
        print("Precision for multi: " + str(precision_multi))
        print("Recall for multi: " + str(recall_multi))
        print("F1 for multi: " + str(f1_multi))
        print("-------------------------------------------")
        print("Single win: " + str(single_win))
        print("Multi win: " + str(multi_win))
        print("Tie: " + str(tie))
        