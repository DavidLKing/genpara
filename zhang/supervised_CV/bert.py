# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 15:42:13 2019

@author: 13982
"""

import pdb
import torch
import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class BertBatch:
    
    def __init__(self, device=-1):
        # for cuda, 0 or greater
        # for cpu, -1
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert.eval()
        self.device = device
        if device >= 0:
            self.bert.to('cuda')
            
    def pad(self, sents):
        lengths = [len(x) for x in sents]
        max_length = max(lengths)
        for seq in sents:
            while len(seq) < max_length:
                # BERT pad char = 0
                seq += [0]
        new_sents = torch.tensor(sents)
        if self.device >= 0:
            new_sents.to('cuda')
        return new_sents
    
    def extract(self, sentences, batchSize):
        # TODO add multithreading/processing
        total = len(sentences) // batchSize
        
        tensors = []
        batchLoc = 0
        # i = 0
        while batchLoc <= total:
            print("On batch", batchLoc, "of", total)
            # print("batchLoc", batchLoc)
            # print("batchLoc * batchSize", batchLoc * batchSize)
            # print("(batchLoc + 1) * batchSize", (batchLoc + 1) * batchSize)
            # print("total sents", len(sentences))
            # pdb.set_trace()
            try:
               toID = sentences[(batchLoc * batchSize) : ((batchLoc + 1) * batchSize)]
            except:
               print("This should be the last batch")
               toID = sentences[(batchLoc * batchSize) : ]
            if toID != []:
               sents = [ 
                   [ tokenizer.vocab.get(
                     x, tokenizer.vocab['[UNK]']
                     ) for x in sent 
                     ] for sent in toID 
                   ]
               # sents = [self.tokenizer.convert_tokens_to_ids(
               #        x
               #    )
               #    for x in toID]
               padded_sents = self.pad(sents)
               if self.device >= 0:
                  padded_sents = padded_sents.to('cuda')
               embeddings, _ = self.bert(padded_sents)
               [tensors.append(np.asarray(x.tolist())) for x in embeddings[0]]
               # pdb.set_trace()
            batchLoc += 1
            
        assert(len(tensors) == len(sentences))
        tensors = np.asarray(tensors)
        # pdb.set_trace()
        return tensors