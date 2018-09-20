import sys
import pdb
from build_phrase_table import PhraseTable

def clean(src_tgt_array):
    out = []
    for line in src_tgt_array:
        if not line[2].startswith("No Alignments"):
            out.append(line)
    return out

p = PhraseTable()
elmos = [x.split('\t') for x in open(sys.argv[1], 'r').readlines()]
elmos = clean(elmos)
phrase_table = p.build(elmos)
for src in phrase_table:
    for tgt in phrase_table[src]:
        print(src, '\t', tgt)