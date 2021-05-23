#!/usr/bin/env python3

import random
import sys
import pdb
from tqdm import tqdm

infile = sys.argv[1]

label_store = {}
label_set = set()

print("Importing lines")
with open(infile, 'r') as infl:
    for line in tqdm(infl):
        if not line.startswith("#START"):
            line = line.split('\t')
            example = line[0].lower()
            label = line[1]
            label_set.add(label)
            if label not in label_store:
                label_store[label] = [label]
            # cut out duplicates
            if example not in label_store[label]:
                label_store[label].append(example)

all_pos = []
all_neg = []

print("Generating positive examples")
for label in tqdm(label_store):
    for first_example in label_store[label]:
        for second_example in label_store[label]:
            if first_example != second_example:
                line = "1\t{} [@] {}\n".format(first_example, second_example)
                inverse_line = "1\t{} [@] {}\n".format(second_example, first_example)
                if line not in all_pos:
                    all_pos.append(line)
                    all_pos.append(inverse_line)

up_to = len(all_pos)
print("Generated {} positive paraphrases pairs for the classifier".format(up_to))

label_list = list(label_set)

progress_bar = tqdm(total = up_to)

while len(all_neg) < up_to:
    first_label = random.choice(label_list)
    second_label = random.choice(label_list)
    if first_label != second_label:
        example_1 = random.choice(label_store[first_label])
        example_2 = random.choice(label_store[second_label])
        line = "0\t{} [@] {}\n".format(example_1, example_2)
        line_inverse = "0\t{} [@] {}\n".format(example_2, example_1)
        if line not in all_neg:
            all_neg.append(line)
            all_neg.append(line_inverse)
            progress_bar.update(2)

progress_bar.close()

print("Generated {} negative examples".format(len(all_neg)))

print("Shuffling")
random.shuffle(all_pos)
random.shuffle(all_neg)

# ADDED LATER
def make_bert(annotation, all_paras, up_to):
    all_paras_4 = []
    while len(all_paras_4) < up_to:
        text_a = random.choice(all_paras).strip().split('\t')[1]
        text_b = random.choice(all_paras).strip().split('\t')[1]
        if text_a != text_b:
            all_paras_4.append("{}\t{} [SEP] {}\n".format(annotation, text_a, text_b))
    return all_paras_4

all_pos = make_bert("1", all_pos, up_to)
all_neg = make_bert("0", all_neg, up_to)

print("Additional mixing to utilize all 4 sentences")

print("Splitting")
def split(dataset):
    # 70, 10, 20 split
    train = dataset[0:int(up_to * 0.7)]
    dev = dataset[int(up_to * 0.7):int(up_to * 0.8)]
    test = dataset[int(up_to * 0.8):]
    return train, dev, test

train_pos, dev_pos, test_pos = split(all_pos)
train_neg, dev_neg, test_neg = split(all_neg)

train = train_pos + train_neg
dev = dev_pos + dev_neg
test = test_pos + test_neg

print("Reshuffling splits")
random.shuffle(train)
random.shuffle(dev)
random.shuffle(test)

print("Writing files")

def writeout(array, data_set_string):
    with open(data_set_string, 'w') as of:
        for line in array:
            of.write(line)

writeout(train, 'paraphrase_classifier_train.tsv')
writeout(dev, 'paraphrase_classifier_dev.tsv')
writeout(test, 'paraphrase_classifier_test.tsv')
