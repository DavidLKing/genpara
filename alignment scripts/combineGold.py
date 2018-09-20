import sys
import os
import pdb
from build_phrase_table import PhraseTable

class Combine:
    def __init__(self):
        pass


    def read_gold(self, line_array, gold_file):
        """
        Read in a file form the gold annotations, strip out the weird formatting, and return lines
        in [source, target, alignment] structures
        :param line_array: an array of lines or an empty array
        :param gold_file: annotations from the goldalignment-repo---https://github.com/ajdagokcen/goldalign-repo.git
        :return: augmented array
        """
        gold_lines = open(gold_file, 'r').readlines()
        gold_lines = [x.split('\t') for x in gold_lines]
        # sanity check to make sure we actually have annotations
        assert([x[7] != '' and x[8] != '' for x in gold_lines])
        # pull source, target, and annotations ONLY if both agree it's a paraphrase
        gold_lines = [[x[1], x[3], ' '.join([x[7], x[8]]).strip()] for x in gold_lines if x[5] == '1' and x[6] == '1']
        # gold_lines = [[x[1], x[2], x[3]] for x in gold_lines if x[5] == '1' or x[6] == '1']
        line_array += gold_lines
        return line_array

if __name__ == '__main__':
    p = PhraseTable()
    c = Combine()
    test = []
    # path = '../data/goldalign-repo/data/users/arbit/demo-user-1_demo-user-2/complete/vpd-corpus/'
    # for files in os.listdir(path):
    for files in sys.argv[1:]:
        # test = c.read_gold(test, path + files)
        test = c.read_gold(test, files)
    phrase_table = p.build(test)
    # for line in test:
    #     print(line)
    for src in phrase_table:
        for tgt in phrase_table[src]:
            print(src, '\t', tgt)
