import sys
import pdb
import random

turns = open('vp16.base.word.shuffled.35.txt', 'r').readlines()

indices = open('vp16.base.shuffled.35.indices', 'r').readlines()

assert(len(turns) == len(indices))

label_counts = {}
label_turns = {}

for line, idx in zip(turns, indices):
    line = line.split('\t')
    label = line[0]
    if label not in label_counts:
        label_counts[label] = 0
    label_counts[label] += 1
    if label not in label_turns:
        label_turns[label] = []
    label_turns[label].append((line[1], idx))

max_val = label_counts[max(label_counts, key=label_counts.get)]

for label in label_turns:
    while len(label_turns[label]) < max_val:
        label_turns[label].append(random.choice(label_turns[label]))

outturns = []


for label in label_turns:
    for turn in label_turns[label]:
        outturns.append([label, turn[0], turn[1]])

random.shuffle(outturns)

outlines = []
outindices = []
for turn in outturns:
    outlines.append('\t'.join([turn[0], turn[1]]))
    outindices.append(turn[2])

outdials = open('vp16.base.word.shuffled.35.balanced.txt', 'w')

outidxs = open('vp16.base.shuffled.35.balanced.indices', 'w')

for line in outlines:
    outdials.write(line)

for idx in outindices:
    outidxs.write(idx)


