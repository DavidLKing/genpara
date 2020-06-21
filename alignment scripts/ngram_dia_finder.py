import sys
import pdb
from normalize import normalize
from rebuild_dialogs import rebuild

# args: corrected.tsv filterend.tsv output.tsv

corrected = open(sys.argv[1], 'r').readlines()

filtered = open(sys.argv[2], 'r')

outfile = open(sys.argv[3], 'w')

n = normalize()
r = rebuild()

dia_num_ids = r.rebuild_dialogs(corrected)

scores = []
srcs = []
paras = []

header = filtered.readline()

for line in filtered:
    line = line.strip().split('\t')
    scores.append(float(line[6]))
    srcs.append(line[1])
    paras.append(line[4])

scores = n.normalize_and_invert(scores)

missing = 0
total = len(srcs)
for src, para, score in zip(srcs, paras, scores):
    if src in dia_num_ids:
        for match in dia_num_ids[src]:
            dia = match[0]
            turn = match[1]
            outfile.write(' \t'.join([str(dia), str(turn), para, str(score)]) + '\n')
    else:
        missing += 1

print("Missing", missing, "of", total)