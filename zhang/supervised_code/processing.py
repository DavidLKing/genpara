# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 17:12:26 2019

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

class DataProcessing:
    
    """Add hyphens between locs of sent"""
    def replaceSpace(self, sent, locs):
        spaceIdxs = []
        for idx, character in enumerate(sent):
            if character is " ":
                spaceIdxs.append(idx)
        newSent = sent
        for loc in locs:
            try:
                toReplace = spaceIdxs[loc]
                newSent = newSent[:toReplace] + "-" + newSent[toReplace+1:]
            except:
                continue
        return newSent
    
    """TO BE TESTED: for each ROOT, check for certain dependencies"""
    def depRootAnalyze(self, sent, alignments, nlp, which='left'):
        ind = []
        paired = []
        doc = nlp(sent)
        s = list(doc.sents)[0]
        allowed_dep = ['prt', 'prep', 'advmod', 'npadvmod']
        for align in alignments:
            if which == 'left':
                paired.append(int(align.split('-')[0]))
            elif which == 'right':
                paired.append(int(align.split('-')[1]))
        root = s.root
        
        for child in root.children:
            """If a child has allowed dependency relationship with the ROOT token, recursively return the branch"""
            if child.dep_ in allowed_dep:
                branch = [root.i] + self.recurse_from_node(child)
                ind.append(branch)
                
        """DEBUGGED: to cater to 'exercise' case, must have at least the root itself in the first place!!"""
        if len(ind) == 0:
            ind = [[root.i]]
        
        """Remove aligned indices from average process"""
        for ii in paired:
            for br in ind:
                if ii in br:
                    br.remove(ii)
        
        for i in ind:
            for j in i:
                print(j)
                print(doc[j])
        
        return ind
    
    """✔Given a single alignment and unpaired indices, return the indices to be averaged"""
    def depAnalyze(self, sent, alignments, nulls, nlp, which='left'):
        ind = []
        paired = []
        doc = nlp(sent)
        allowed_dep = ['npadvmod', 'advmod', 'acomp']
        for align in alignments:
            if which == 'left':
                paired.append(int(align.split('-')[0]))
            elif which == 'right':
                paired.append(int(align.split('-')[1]))
        for i in paired:
            token = doc[i]
            for j in nulls:
                child = doc[j]
                """If a child has allowed dependency relationship with a paired parent, recursively return the branch"""
                if child.text in [t.text for t in token.children] and child.dep_ in allowed_dep:
                    branch = [i] + self.recurse_from_node(child)
                    ind.append(branch)
                    
        return ind
    
    """✔Return a dict of tuples[index] = (POS, DEP, text), given a list of indices"""
    def get_POS_and_DEP(self, sent, nlp, index):
        tuples = {}
        doc = nlp(sent)
        for i in index:
            token = doc[i]
            tuples[i] = (token.pos_, token.dep_, token.text)
        return tuples

    """Return list of POS and DEP individually in the order of alignment indices"""
    def get_POS(self, tuples, index):
        pos = []
        for i in index:
            pos.append(str(tuples[i][0]))
        return pos

    def get_DEP(self, tuples, index):
        dep = []
        for i in index:
            dep.append(str(tuples[i][1]))
            
        return dep

    """Given two tuples of training info, return a string of formatted output"""
    """'-' separates sentences, ';' separates index, ',' separates contents"""
    """换一个output体现形式！！！！"""
    def format_tuples(self, tuple1, tuple2):
        output = ''
        for i, tup in enumerate(tuple1):
            output += str(tup)
            if i < len(tuple1)-1:
                output += ';'
        output += '-'
        for i, tup in enumerate(tuple2):
            output += str(tup)
            if i < len(tuple2)-1:
                output += ';'
        return output

    """Recursively read from a node and returns the whole branch indices until the end"""
    def recurse_from_node(self, token):
        branch = [token.i]
        if token.children is not None:
            for ch in token.children:
                branch += self.recurse_from_node(ch)
        return branch

    """FOLLOWING FUNCTION TO BE TESTED!!!!"""
    def recurse_POS_and_DEP(self, token):
        branch = [(token.i, token.pos_, token.dep_)]
        if token.children is not None:
            for ch in token.children:
                branch += self.recurse_from_node(ch)
        return branch
    
    """Return True/1 if in gold, False/0 otherwise [Return True only if the whole alignments are true, not allow a single error]"""
    def is_gold(self, align, gold):
        if '----' in align or '====' in align:
            a = []
            if '----' in align:
                a = align.split('----')
            elif '====' in align:
                a = align.split('====')
            """Reprocess the strings back to the list"""
            left = a[0][1:-1].split(', ')
            right = a[1][1:-1].split(', ')
            for l in left:
                for r in right:
                    b = str(l) + '-' + str(r)
                    if b not in gold:
                        return False
        else:
            if align not in gold:
                return False
            
        return True

    """Given two lists of indices, average the scores and return"""
    """Indice format: l_ind = [4, 5, 6]; r_ind = [2, 4, 5]..."""
    def avgScore(self, l_ind, r_ind, left, right, sent1, sent2):
        l_vec = [left[l] for l in l_ind]
        r_vec = [right[r] for r in r_ind]
        shape = left[0].shape
        
        l_sum = np.zeros((shape))
        r_sum = np.zeros((shape))
        for i in l_vec:
            l_sum += i
        for i in r_vec:
            r_sum += i
            
        l_avg = (l_sum) / len(l_ind)
        r_avg = (r_sum) / len(r_ind)
        
        score = distance.cosine(l_avg, r_avg)
        print("Currently examine: ")
        print(str(l_ind) + '----' + str(r_ind))
        print(str([sent1[i] for i in l_ind]) + " | " + str([sent2[i] for i in r_ind]))
        print("Score is: ")
        print(score)
        
        return score
        
    """Given two lists of indices, average the embeddings and return"""
    """Indice format: l_ind = [4, 5, 6]; r_ind = [2, 4, 5]..."""
    def avgEmbedding(self, l_ind, r_ind, left, right, sent1, sent2, which='left'):
        l_vec = [left[l] for l in l_ind]
        r_vec = [right[r] for r in r_ind]
        shape = left[0].shape
        
        """Using numpy to perform vector addition"""
        l_sum = np.zeros((shape))
        r_sum = np.zeros((shape))
        for i in l_vec:
            l_sum += i
        for i in r_vec:
            r_sum += i
            
        l_avg = (l_sum) / len(l_ind)
        r_avg = (r_sum) / len(r_ind)
        
#        print("Currently examine: ")
#        print(str(l_ind) + '----' + str(r_ind))
#        print(str([sent1[i] for i in l_ind]) + " | " + str([sent2[i] for i in r_ind]))
#        print("Get the avg embeddings!")
        
        if which == 'left':
            return l_avg
        elif which == 'right':
            return r_avg
        
    """return一对包含了所有没被aligned的index！！！！"""
    def hasNull(self, alignments, lNumWords, rNumWords):
        """包含了所有word的index的set！！！！"""
        lSet = set(range(lNumWords))
        rSet = set(range(rNumWords))
        """对于每一对index-alignments"""
        for pair in alignments:
            pair = pair.split('-')
            lVal = int(pair[0])
            rVal = int(pair[1])
            """把已经aligned了的index从set中去掉！！！！"""
            if lVal in lSet:
                lSet.remove(lVal)
            if rVal in rSet:
                rSet.remove(rVal)
        lSet = list(lSet)
        rSet = list(rSet)
        return(lSet, rSet)
    
    """将已有的两个句子所有的index-alignment转换为word-word的alignment！！！！"""
    def alignsToRead(self, aligns, s1, s2):
        readable = []
        newAligns = []
        for alignment in aligns:
            """将alignment根据 - 分成两部分，左句和右句"""
            vals = alignment.split('-')
            left = int(vals[0])
            right = int(vals[1])
            toPrint = s1[left] + " | " + s2[right]
            newAligns.append(alignment)
            readable.append(toPrint)
        return (newAligns, readable)
    
    def set_custom_boundaries(self, doc):
        for token in doc[1:]:
            doc[token.i].is_sent_start = False
        return doc
    
    """Get the distance of indices of aligned pairs normalized with the longer of the sentence length"""
    def getIndexDistanceNormalized(self, aligns, sent_len):
        if '====' in aligns or '----' in aligns:
            align = ''
            if '====' in aligns:
                align = aligns.split('====')
            elif '----' in aligns:
                align = aligns.split('----')
            left = align[0][1:-1].split(', ')
            left = [int(i) for i in left]
            right = align[1][1:-1].split(', ')
            right = [int(i) for i in right]
            lsum = np.mean(left)
            rsum = np.mean(right)
            return abs(lsum - rsum) / sent_len
        
        else:
            align = aligns.split('-')
            left = int(align[0])
            right = int(align[1])
            return abs(left - right) / sent_len
    
    """Get the span of alignment candidate for each sentence"""
    def getAlignmentSpan(self, aligns, sent_len, which='left'):
        if '====' in aligns or '----' in aligns:
            align = ''
            if '====' in aligns:
                align = aligns.split('====')
            elif '----' in aligns:
                align = aligns.split('----')
                
            ind = []
            if which == 'left':
                left = align[0][1:-1].split(', ')
                ind = [int(i) for i in left]
            elif which == 'right':
                right = align[1][1:-1].split(', ')
                ind = [int(i) for i in right]
            
            max_ind = max(ind)
            min_ind = min(ind)
            return (max_ind - min_ind) / sent_len
    
        else:
            return 0

    """Check if the core of the aligned pairs have the same dependency"""
    def hasSameDep(self, line):
        pass

    """Get the gold label"""
    def getGold(self, line):
        if line[4] == 'True':
            return 1
        elif line[4] == 'False':
            return 0
        return None
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


