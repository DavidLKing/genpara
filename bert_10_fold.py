#!/usr/bin/env python3

import pdb
import random
import os
import shutil
import nltk
import argparse
import tqdm
import os
import multiprocessing as mp
import subprocess
from nltk.translate import bleu
from nltk.translate.bleu_score import SmoothingFunction


class TenFold:

    def __init__(self):
        self.folds = ['0', '1', '2',
                      '3', '4', '5',
                      '6', '7', '8',
                      '9']
        self.smoothing = SmoothingFunction().method4

    def rank_and_extrace(self, possibles, reverse=False):

        total = len(possibles) #[label])

        print("Scoring")
        scored_labels = {}
        for label in tqdm.tqdm(possibles, total=total):
            threads = mp.cpu_count() - 1
            p = mp.Pool(threads)
            scored = []
            # pdb.set_trace()
            for pos in p.imap_unordered(self.rank_job, possibles[label]):
            # for pos in possibles:
                if pos != None:
                    # pdb.set_trace()
                    scored.append(pos)
            p.close()
            p.join()

            print("Sorting")

            sorted(scored, key=lambda x: x[0])
            if reverse:
                scored.reverse()
            scored_labels[label] = [x[1] for x in scored]
        return scored_labels


    def rank_job(self, pos):
        # self.prog_bar.update(1)
        try:
            # pdb.set_trace()
            score = bleu(pos[0], pos[1], smoothing_function=self.smoothing)
            return score, pos
        except:
            pass

    def gen_pos(self, labels):
        # Enumerate all possible positive examples:
        positives = {}
        for label in labels:
            for sent1 in labels[label]:
                for sent2 in labels[label]:
                    if sent1 != sent2:
                        if label not in positives:
                            positives[label] = []
                        positives[label].append((sent1, sent2, label, label, 1))
                        positives[label].append((sent2, sent1, label, label, 1))
        positives = self.rank_and_extrace(positives, reverse=False)
        return positives

    def gen_neg(self, labels):
        # Enumerate all possible negative examples:
        negatives = {}
        for label1 in labels:
            for label2 in labels:
                if label1 != label2:
                    if label1 not in negatives:
                        negatives[label1] = []
                    if label2 not in negatives:
                        negatives[label2] = []
                    for sent1 in labels[label1]:
                        for sent2 in labels[label2]:
                            if sent1 != sent2:
                                negatives[label1].append((sent1, sent2, label1, label2, 0))
                                negatives[label2].append((sent2, sent1, label2, label1, 0))
        # TEMP HACK FOR PROTOTYPING
        # random.shuffle(negatives)
        # negatives = negatives[0:378756]
        negatives = self.rank_and_extrace(negatives, reverse=True)
        return negatives

    def gen_paras(self, extracted):
        pass

    def extract(self, corrected_file):
        labels = {}
        with open(corrected_file, 'r')as infile:
            for line in infile:
                if not line.startswith('#'):
                    line = line.strip().split('\t')
                    question = line[0]
                    label = line[1]
                    assert(label != '<None>')
                    if label not in labels:
                        labels[label] = []
                        labels[label].append(label)
                    labels[label].append(question)
        return labels

    def split(self, data, folds):
        splits = {}
        fold_num = len(data) // len(folds)
        # not really required
        # remainder = len(data) % len(folds)
        curr_num = 0
        for f in folds:
            dev_start = curr_num * fold_num
            dev_end = (curr_num + 1) * fold_num
            dev = data[dev_start:dev_end]
            train = data[0:dev_start] + data[dev_end:]
            splits[f] = {}
            splits[f]['train'] = train
            splits[f]['dev'] = dev
            curr_num += 1
        return splits

    def split_by_label(self, data, folds, labels):
        splits = {}
        curr_num = 0
        for fold in labels:
            dev = [x for x in data if x[2] in fold]
            train = [x for x in data if x[2] not in fold]
            splits[curr_num] = {}
            splits[curr_num]['train'] = train
            splits[curr_num]['dev'] = dev
            curr_num += 1
        return splits

    def write_out(self, outfile, data):
        idx1 = 0
        idx2 = 1
        with open(outfile, 'w') as of:
            # header = ['index', 'sentence1', 'sentence2', 'label']
            header = ['quality', 'id 1', 'id 2', 'sentence 1', 'sentence 2']
            of.write('\t'.join(header) + '\n')
            for line in data:
                # outline = [str(idx), line[0], line[1], str(line[-1])]
                # original - 1324
                # outline = [str(line[-1]), str(idx1), str(idx2), line[0] + ' [+] ' + line[2], line[1] + ' [+] ' + line[3]]
                # 1234
                # outline = [str(line[-1]), str(idx1), str(idx2), line[0] + ' [+] ' + line[1], line[2] + ' [+] ' + line[3]]
                # 34, mrpc
                outline = [str(line[-1]), str(idx1), str(idx2), line[0], line[1]]
                of.write('\t'.join(outline) + '\n')
                idx1 += 2
                idx2 += 2

    def get_top(self, positives, negatives, limit=25):
        # knarley
        # minimum = min(
        #     [
        #         min([len(positives[x]) for x in positives]),
        #         min([len(negatives[x]) for x in negatives])
        #     ]
        # )
        # new_pos = {}
        # new_neg = {}
        data = []
        assert(len(list(positives.keys())) <= len(list(negatives.keys())))
        for label in positives:
            minimum = min(
                [
                    len(positives[label]),
                    len(negatives[label])
                ]
            )
            # assert(label not in new_neg)
            # assert(label not in new_pos)
            # new_pos[label] = positives[0:minimum]
            # new_neg[label] = negatives[0:minimum]
            for example in positives[label][0:minimum]:
                data.append(example)
            for example in negatives[label][0:minimum]:
                data.append(example)

        # pdb.set_trace()
        # pull 10% # 25%
        # limit = int(len(positives) / 10)
        # positives = positives[0:limit]

        return data

    def main(self):
        parser = argparse.ArgumentParser()

        ## Required parameters
        parser.add_argument("--data",
                            default=None,
                            type=str,
                            required=True,
                            help="The input data file. Something like 'corrected.tsv'")

        parser.add_argument("--out",
                            default=None,
                            type=str,
                            required=True,
                            help="output directory (overwrites everything)")
        args = parser.parse_args()

        if not os.path.isdir(args.out):
            os.mkdir(args.out)
        else:
            for fold in os.listdir(args.out):
                # os.rmdir(os.path.join(args.out, fold))
                shutil.rmtree(os.path.join(args.out, fold))
        for fold in self.folds:
            os.mkdir(os.path.join(args.out, fold))

        # 10 fold by label
        groups = self.extract(args.data)
        just_labels = list(groups.keys())
        folds = len(self.folds)
        num = round(len(just_labels) / folds)
        splits = [just_labels[i * num:(i + 1) * num] for i in range(folds)]
        positives = self.gen_pos(groups)
        negatives = self.gen_neg(groups)

        data = self.get_top(positives, negatives, limit=10)

        # shuffle and split
        # data = positives + negatives
        random.shuffle(data)

        # folds = self.split(data, self.folds)
        folds = self.split_by_label(data, self.folds, splits)

        print("Writing out folds")

        for fold in folds:
            train_file = os.path.join(args.out, str(fold), 'train.tsv')
            dev_file = os.path.join(args.out, str(fold), 'dev.tsv')
            self.write_out(train_file, folds[fold]['train'])
            self.write_out(dev_file, folds[fold]['dev'])

        # self.gen_paras(groups)



if __name__ == '__main__':
    ten = TenFold()
    ten.main()
