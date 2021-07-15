import sys
import pdb
from tqdm import tqdm
import random
import pandas as pd

commons = pd.read_excel('Common Pod Question Labels.xlsx', sheet_name=None, engine='openpyxl')
extras  = pd.read_excel('corrected_data_june_october.xlsx', sheet_name=None, engine='openpyxl')

questions = []
labels = []
numeric_labels = {}
answers = []

for sheet in commons:
    sheet_labels = commons[sheet]['Label'].values.tolist()
    sheet_labels_numeric = commons[sheet]['Numerical Label'].values.tolist()
    sheet_questions = commons[sheet]['Questions'].values.tolist()
    sheet_answers = commons[sheet]['Answer'].values.tolist()

    for num, label in zip(sheet_labels_numeric, sheet_labels):
        if label not in numeric_labels:
            numeric_labels[label] = num
        assert(numeric_labels[label] == num)

    for ques, lab, ans in zip(sheet_questions, sheet_labels, sheet_answers):
        questions.append(ques)
        labels.append(lab)
        answers.append(ans)

missing = 0
for month in extras:
    month_labels = extras[month]['Label'].values.tolist()
    month_questions = extras[month]['Question'].values.tolist()
    month_answers = extras[month]['Answer'].values.tolist()
    for ques, lab, ans in zip(month_questions, month_labels, month_answers):
        if lab in numeric_labels:
            questions.append(ques)
            labels.append(lab)
            answers.append(ans)
        else:
            missing += 1
print("Threw out {} data point for not having numbered labels".format(missing))

gen_line = lambda ques, lab, answ: '{}\t{}\t{}\t{}\tFalse\tFF000000\tFF000000\n'.format(ques,lab,answ,lab)

lines = []

for ques, lab, answ in zip(questions, labels, answers):
    lines.append(gen_line(ques, lab, answ))

random.shuffle(lines)

with open('cosi_corrected.tsv', 'w') as outfile:
    for line in tqdm(lines):
        outfile.write(line)
