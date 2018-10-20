from allennlp.modules.elmo import Elmo, batch_to_ids
import numpy as np
import pdb
from scipy.spatial import distance
import csv
import sys
import networkx as nx

def resultsToAlignments(results):
    alignments = []
    for element in results:
        if "a" in element[0]:
            left = element[0][-1]
            right = element[1][-1]
        else:
            left = element[1][-1]
            right = element[0][-1]
        outString = str(left) + "-" + str(right)
        alignments.append(outString)
    return alignments

def alignsToRead(aligns, s1, s2):
    readable = []
    for alignment in aligns:
        left = int(alignment[0])
        right = int(alignment[-1])
        toPrint = s1[left] + " | " + s2[right]
        readable.append(toPrint)
    return readable

#options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
options_file = "defaultOptions.json"
#weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
weight_file = "defaultWeights.hdf5"

elmo = Elmo(options_file, weight_file, 1)

truePos = 0
precBtm = 0
recBtm = 0

sentSet = set()
sentences = []
goldScores = []
with open(sys.argv[1]) as tsv:
    for line in csv.reader(tsv, dialect="excel-tab"):
        combo = " ".join(line)
        if combo not in sentSet:
            #pdb.set_trace()
            goldScores.append(set(line[2].split()))
            sent1 = line[0].split()
            sent2 = line[1].split()
            sentences.append(sent1)
            sentences.append(sent2)
            sentSet.add(combo)

threshold = 0.7
batchSize = 128
batchLoc = 0
i = 0
while batchLoc < len(sentences):
    toID = sentences[batchLoc : (batchLoc + batchSize)]
    character_ids = batch_to_ids(toID)

    embeddings = elmo(character_ids)

    npTensor = embeddings['elmo_representations'][0].detach().numpy()
    j = 0
    while (j+1) < len(npTensor):
        G = nx.Graph()
        sent1 = sentences[i]
        sent2 = sentences[i+1]
        left = npTensor[j]
        right = npTensor[j+1]
        numWords = 0
        for idx, word in enumerate(left):
            if np.count_nonzero(word) or np.count_nonzero(right[idx]):
                numWords += 1
        scores = np.ones((numWords, numWords))
        for row in range(numWords):
            lVec = left[row]
            for column in range(numWords):
                rVec = right[column]
                if np.count_nonzero(lVec) and np.count_nonzero(rVec):
                    wordA = "a_" + str(row)
                    wordB = "b_" + str(column)
                    dist = distance.cosine(lVec, rVec)
                    if dist >= threshold:
                        dist = 0
                    else:
                        dist = 1-dist
                        G.add_edge(wordA, wordB, weight=dist)

        results = nx.max_weight_matching(G, maxcardinality=True)
        alignments = resultsToAlignments(results)
        alignsRead = alignsToRead(alignments, sent1, sent2)

        #pdb.set_trace()
        gold = goldScores[int(i/2)]
        precBtm += len(alignments)
        recBtm += len(gold)
        for pair in alignments:
            if pair in gold:
                truePos += 1
        print(int(i/2))
        print(alignments)
        print(alignsRead)
        print()
        i += 2
        j += 2
        #pdb.set_trace()
    batchLoc += batchSize

precision = truePos/float(precBtm)
recall = truePos/float(recBtm)
f1 = 2*((precision*recall)/(precision + recall))
print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("F1: " + str(f1))