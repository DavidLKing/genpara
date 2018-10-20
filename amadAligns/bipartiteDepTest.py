from allennlp.modules.elmo import Elmo, batch_to_ids
import numpy as np
import pdb
from scipy.spatial import distance
import csv
import sys
import networkx as nx
import depParse as deps

def hasNull(results, nodes):
    for pair in results:
        for node in pair:
            if node in nodes:
                nodes.remove(node)
    return nodes

def getScore(G, results):
    score = 0
    for pair in results:
        val = G[pair[0]][pair[1]]['weight']
        score += val
    return score

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

def getIdxMap(sent):
    offset = 0
    idxMap = dict()
    for idx, token in enumerate(sent):
        idxMap[idx] = set()
        loc = idx + offset
        idxMap[idx].add(loc)
        if '-' in token:
            for letter in token:
                if letter == "-":
                    offset += 1
                    loc = idx + offset
                    idxMap[idx].add(loc)
    return idxMap

def alignsToRead(aligns, s1, s2):
    readable = []
    newAligns = []
    map1 = getIdxMap(s1)
    map2 = getIdxMap(s2)
    for alignment in aligns:
        left = int(alignment[0])
        right = int(alignment[-1])
        for lVal in map1[left]:
            for rVal in map2[right]:
                alignString = str(lVal) + "-" + str(rVal)
                newAligns.append(alignString)
        toPrint = s1[left] + " | " + s2[right]
        readable.append(toPrint)
    return (readable, newAligns)

def main():
    #options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    options_file = "defaultOptions.json"
    #weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
    weight_file = "defaultWeights.hdf5"

    elmo = Elmo(options_file, weight_file, 1)

    trials = []
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
                trials.append([])

    threshold = 0.8
    batchSize = 128

    for k in range(2):
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
                nodes = set()
                sent1 = sentences[i]
                sent2 = sentences[i+1]
                left = npTensor[j]
                right = npTensor[j+1]
                numWords = 0
                for idx, word in enumerate(left):
                    if np.count_nonzero(word) or np.count_nonzero(right[idx]):
                        numWords += 1
                for row in range(numWords):
                    lVec = left[row]
                    for column in range(numWords):
                        rVec = right[column]
                        if np.count_nonzero(lVec) and np.count_nonzero(rVec):
                            wordA = "a_" + str(row)
                            wordB = "b_" + str(column)
                            nodes.add(wordA)
                            nodes.add(wordB)
                            dist = distance.cosine(lVec, rVec)
                            if dist < threshold:
                                dist = 1-dist
                                G.add_edge(wordA, wordB, weight=dist)

                results = nx.max_weight_matching(G, maxcardinality=True)
                alignments = resultsToAlignments(results)
                allAligns = alignsToRead(alignments, sent1, sent2)
                alignsRead = allAligns[0]
                alignments = allAligns[1]
                nulls = hasNull(results, nodes)
                score = getScore(G, results)
                trials[int(i/2)].append((alignments, alignsRead, score))

                hyphenA = []
                hyphenB = []
                for null in nulls:
                    if 'a' in null:
                        hyphenA.append(int(null.split('_')[1]))
                    else:
                        hyphenB.append(int(null.split('_')[1]))
                
                raw1 = " ".join(sent1)
                raw2 = " ".join(sent2)
                new1 = deps.hyphenate(raw1, hyphenA)
                new2 = deps.hyphenate(raw2, hyphenB)
                sentences[i] = new1.split()
                sentences[i+1] = new2.split()

                i += 2
                j += 2
            batchLoc += batchSize


    truePos = 0
    precBtm = 0
    recBtm = 0
    for i, pair in enumerate(trials):
        #ranked = sorted(pair, key=lambda x: x[2], reverse=True)
        #best = ranked[1]
        best = pair[1]
        alignments = best[0]
        alignsRead = best[1]

        gold = goldScores[i]
        precBtm += len(alignments)
        recBtm += len(gold)
        for pair in alignments:
            if pair in gold:
                truePos += 1
        print(i)
        print(alignments)
        print(alignsRead)
        print()
    
    precision = truePos/float(precBtm)
    recall = truePos/float(recBtm)
    f1 = 2*((precision*recall)/(precision + recall))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("F1: " + str(f1))


if __name__ == "__main__":
    main()