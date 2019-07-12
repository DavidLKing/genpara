
# coding: utf-8

# In[1]:


import subprocess
import pdb
import pandas
import numpy as np
import sys
from sklearn.metrics import average_precision_score

# In[2]:


import os; os.getcwd()


# In[19]:


# feats = open('sanity.feats', 'r').readlines()
feats = open(sys.argv[1],  'r').readlines()


# In[20]:


# feats


# In[21]:


# glove_src_para_sim


# In[22]:


def normalize_and_invert(dist):
    max = 0
    min = 1
    for float_value in dist:
        if float_value > max:
            max = float_value
        elif float_value < min:
            min = float_value
    # normalize
    new_dist = []
    for old_score in dist:
        new_score = (old_score - min) / (max - min)
        # invert
        new_dist.append(1 - new_score)
    # pdb.set_trace()
    return new_dist

def normalize(dist):
    max = 0
    min = 1
    for float_value in dist:
        if float_value > max:
            max = float_value
        elif float_value < min:
            min = float_value
    # normalize
    new_dist = []
    for old_score in dist:
        new_score = (old_score - min) / (max - min)
        # invert
        new_dist.append(new_score)
    # pdb.set_trace()
    return new_dist


# temp_scores = [x.split('\t')[1].split(' ') for x in feats]

scored_csv = pandas.read_csv(sys.argv[1], delimiter='\t')

#ntemp_labels = [x.split('\t')[0] for x in feats]
temp_labels = scored_csv['annotation'].tolist()

dialog = [x for x in scored_csv['dialog'].tolist()]
turn = [x for x in scored_csv['turn'].tolist()]
glove_src_para_sim = [x for x in scored_csv['glove_src_para_sim'].tolist()]
glove_src_para_dist = [x for x in scored_csv['glove_src_para_dist'].tolist()]
glove_src_para_david = [x for x in scored_csv['glove_src_para_david'].tolist()]
glove_src_orig_sim = [x for x in scored_csv['glove_src_orig_sim'].tolist()]
glove_src_orig_dist = [x for x in scored_csv['glove_src_orig_dist'].tolist()]
glove_src_orig_david = [x for x in scored_csv['glove_src_orig_david'].tolist()]
glove_orig_para_sim = [x for x in scored_csv['glove_orig_para_sim'].tolist()]
glove_orig_para_dist = [x for x in scored_csv['glove_orig_para_dist'].tolist()]
glove_orig_para_david = [x for x in scored_csv['glove_orig_para_david'].tolist()]
glove_align_para_sim = [x for x in scored_csv['glove_align_para_sim'].tolist()]
glove_align_para_dist = [x for x in scored_csv['glove_align_para_dist'].tolist()]
glove_align_para_david = [x for x in scored_csv['glove_align_para_david'].tolist()]
w2v_src_para_sim = [x for x in scored_csv['w2v_src_para_sim'].tolist()]
w2v_src_para_dist = [x for x in scored_csv['w2v_src_para_dist'].tolist()]
w2v_src_para_david = [x for x in scored_csv['w2v_src_para_david'].tolist()]
w2v_src_orig_sim = [x for x in scored_csv['w2v_src_orig_sim'].tolist()]
w2v_src_orig_dist = [x for x in scored_csv['w2v_src_orig_dist'].tolist()]
w2v_src_orig_david = [x for x in scored_csv['w2v_src_orig_david'].tolist()]
w2v_orig_para_sim = [x for x in scored_csv['w2v_orig_para_sim'].tolist()]
w2v_orig_para_dist = [x for x in scored_csv['w2v_orig_para_dist'].tolist()]
w2v_orig_para_david = [x for x in scored_csv['w2v_orig_para_david'].tolist()]
w2v_align_para_sim = [x for x in scored_csv['w2v_align_para_sim'].tolist()]
w2v_align_para_dist = [x for x in scored_csv['w2v_align_para_dist'].tolist()]
w2v_align_para_david = [x for x in scored_csv['w2v_align_para_david'].tolist()]
elmo_src_para_sim = [x for x in scored_csv['elmo_src_para_sim'].tolist()]
elmo_src_para_dist = [x for x in scored_csv['elmo_src_para_dist'].tolist()]
elmo_src_para_david = [x for x in scored_csv['elmo_src_para_david'].tolist()]
elmo_src_orig_sim = [x for x in scored_csv['elmo_src_orig_sim'].tolist()]
elmo_src_orig_dist = [x for x in scored_csv['elmo_src_orig_dist'].tolist()]
elmo_src_orig_david = [x for x in scored_csv['elmo_src_orig_david'].tolist()]
elmo_orig_para_sim = [x for x in scored_csv['elmo_orig_para_sim'].tolist()]
elmo_orig_para_dist = [x for x in scored_csv['elmo_orig_para_dist'].tolist()]
elmo_orig_para_david = [x for x in scored_csv['elmo_orig_para_david'].tolist()]
elmo_align_para_sim = [x for x in scored_csv['elmo_align_para_sim'].tolist()]
elmo_align_para_dist = [x for x in scored_csv['elmo_align_para_dist'].tolist()]
elmo_align_para_david = [x for x in scored_csv['elmo_align_para_david'].tolist()]
elmo_sims = [x for x in scored_csv['elmo_sims'].tolist()]
elmo_dist = [x for x in scored_csv['elmo_dist'].tolist()]
elmo_david = [x for x in scored_csv['elmo_david'].tolist()]
bert_src_para_sim = [x for x in scored_csv['bert_src_para_sim'].tolist()]
bert_src_para_dist = [x for x in scored_csv['bert_src_para_dist'].tolist()]
bert_src_para_david = [x for x in scored_csv['bert_src_para_david'].tolist()]
bert_src_orig_sim = [x for x in scored_csv['bert_src_orig_sim'].tolist()]
bert_src_orig_dist = [x for x in scored_csv['bert_src_orig_dist'].tolist()]
bert_src_orig_david = [x for x in scored_csv['bert_src_orig_david'].tolist()]
bert_orig_para_sim = [x for x in scored_csv['bert_orig_para_sim'].tolist()]
bert_orig_para_dist = [x for x in scored_csv['bert_orig_para_dist'].tolist()]
bert_orig_para_david = [x for x in scored_csv['bert_orig_para_david'].tolist()]
bert_align_para_sim = [x for x in scored_csv['bert_align_para_sim'].tolist()]
bert_align_para_dist = [x for x in scored_csv['bert_align_para_dist'].tolist()]
bert_align_para_david = [x for x in scored_csv['bert_align_para_david'].tolist()]
bert_sims = [x for x in scored_csv['bert_sims'].tolist()]
bert_dist = [x for x in scored_csv['bert_dist'].tolist()]
bert_david = [x for x in scored_csv['bert_david'].tolist()]
ng_src_para = [x for x in scored_csv['ng_src_para'].tolist()]
ng_src_orig = [x for x in scored_csv['ng_src_orig'].tolist()]
ng_orig_para = [x for x in scored_csv['ng_orig_para'].tolist()]
ng_align_para = [x for x in scored_csv['ng_align_para'].tolist()]
ng_sum = [x for x in scored_csv['ng_sum'].tolist()]
src = [x for x in scored_csv['src'].tolist()]
align = [x for x in scored_csv['align'].tolist()]
para = [x for x in scored_csv['para'].tolist()]
orig = [x for x in scored_csv['orig'].tolist()]
labels = [x for x in scored_csv['label'].tolist()]








metrics = [(normalize(glove_src_para_sim), 'glove_src_para_sim'),
    (normalize_and_invert(glove_src_para_dist), 'glove_src_para_dist'),
    (normalize_and_invert(glove_src_para_david), 'glove_src_para_joint'),
    (normalize(glove_src_orig_sim), 'glove_src_orig_sim'),
    (normalize_and_invert(glove_src_orig_dist), 'glove_src_orig_dist'),
    (normalize_and_invert(glove_src_orig_david), 'glove_src_orig_joint'),
    (normalize(glove_orig_para_sim), 'glove_orig_para_sim'),
    (normalize_and_invert(glove_orig_para_dist), 'glove_orig_para_dist'),
    (normalize_and_invert(glove_orig_para_david), 'glove_orig_para_joint'),
    (normalize(glove_align_para_sim), 'glove_align_para_sim'),
    (normalize_and_invert(glove_align_para_dist), 'glove_align_para_dist'),
    (normalize_and_invert(glove_align_para_david), 'glove_align_para_joint'),
    (normalize(w2v_src_para_sim), 'w2v_src_para_sim'),
    (normalize_and_invert(w2v_src_para_dist), 'w2v_src_para_dist'),
    (normalize_and_invert(w2v_src_para_david), 'w2v_src_para_joint'),
    (normalize(w2v_src_orig_sim), 'w2v_src_orig_sim'),
    (normalize_and_invert(w2v_src_orig_dist), 'w2v_src_orig_dist'),
    (normalize_and_invert(w2v_src_orig_david), 'w2v_src_orig_joint'),
    (normalize(w2v_orig_para_sim), 'w2v_orig_para_sim'),
    (normalize_and_invert(w2v_orig_para_dist), 'w2v_orig_para_dist'),
    (normalize_and_invert(w2v_orig_para_david), 'w2v_orig_para_joint'),
    (normalize(w2v_align_para_sim), 'w2v_align_para_sim'),
    (normalize_and_invert(w2v_align_para_dist), 'w2v_align_para_dist'),
    (normalize_and_invert(w2v_align_para_david), 'w2v_align_para_joint'),
    (normalize(elmo_src_para_sim), 'elmo_src_para_sim'),
    (normalize_and_invert(elmo_src_para_dist), 'elmo_src_para_dist'),
    (normalize_and_invert(elmo_src_para_david), 'elmo_src_para_joint'),
    (normalize(elmo_src_orig_sim), 'elmo_src_orig_sim'),
    (normalize_and_invert(elmo_src_orig_dist), 'elmo_src_orig_dist'),
    (normalize_and_invert(elmo_src_orig_david), 'elmo_src_orig_joint'),
    (normalize(elmo_orig_para_sim), 'elmo_orig_para_sim'),
    (normalize_and_invert(elmo_orig_para_dist), 'elmo_orig_para_dist'),
    (normalize_and_invert(elmo_orig_para_david), 'elmo_orig_para_joint'),
    (normalize(elmo_align_para_sim), 'elmo_align_para_sim'),
    (normalize_and_invert(elmo_align_para_dist), 'elmo_align_para_dist'),
    (normalize_and_invert(elmo_align_para_david), 'elmo_align_para_joint'),
    (normalize(elmo_sims), 'elmo_sims'),
    (normalize_and_invert(elmo_dist), 'elmo_dist'),
    (normalize_and_invert(elmo_david), 'elmo_joint'),
    (normalize_and_invert(bert_orig_para_dist), 'bert_orig_para_dist'),
    (normalize_and_invert(bert_orig_para_david), 'bert_orig_para_joint'),
    (normalize(bert_align_para_sim), 'bert_align_para_sim'),
    (normalize_and_invert(bert_align_para_dist), 'bert_align_para_dist'),
    (normalize_and_invert(bert_align_para_david), 'bert_align_para_joint'),
    (normalize(bert_sims), 'bert_sims'),
    (normalize_and_invert(bert_dist), 'bert_dist'),
    (normalize_and_invert(bert_david), 'bert_joint'),
    # (normalize_and_invert(ng_src_para), 'ng_src_para'),
    # (normalize_and_invert(ng_src_orig), 'ng_src_orig'),
    # (normalize_and_invert(ng_orig_para), 'ng_orig_para'),
    # (normalize_and_invert(ng_align_para), 'ng_align_para'),
    # (normalize_and_invert(ng_sum), 'ng_sum'),]
    (normalize(ng_src_para), 'ng_src_para'),
    (normalize(ng_src_orig), 'ng_src_orig'),
    (normalize(ng_orig_para), 'ng_orig_para'),
    (normalize(ng_align_para), 'ng_align_para'),
    (normalize(ng_sum), 'ng_sum'),]


def prec(data):
    total = 0
    num = 0
    for datum in data:
        total += 1
        if datum[0] == 1:
            num += 1
    return num / total

def all_pos(data):
    total = 0
    for datum in data:
        if datum[0] == 1:
            total += 1
    return total

def rec(data, denom):
    num = 0
    for datum in data:
        if datum[0] == 1:
            num += 1
    return num / denom

def f1(prec, rec):
    return 2 * ( (prec * rec) / (prec + rec) )

def get_map(annos):
    aps = []
    for idx in range(len(annos)):
        correct = []
        correct_denom = 0
        for denoms in range(idx + 1):
            if annos[denoms] == 0:
                ap = 0.0
            else:
                ap = sum(annos[0: denoms + 1]) / (denoms + 1)
                correct_denom += 1
            correct.append(ap)
            # print('idx', idx, 'denoms', denoms)
            # print("adding", annos[0: denoms + 1], '/', denoms + 1)
            # print('ap', ap)
            # print('correct', correct)
            # print('aps', aps)
        if correct_denom == 0:
            assert(sum(correct) == 0)
            aps.append(0.0)
        else:
            # aps.append(sum(correct) / correct_denom)
            aps.append(sum(correct) / len(correct))
    meanap = sum(aps) / len(annos)
    return meanap

header = ["metric", "percent", "prec", "rec", "f1", "AveP", "MAP"]
print('\t'.join(header))
label_names = set(labels)
for met in metrics:
    tupes = []
    met_label = met[1]
    for anno, score in zip(temp_labels, met[0]):
        tupes.append((anno, score))
    tupes.sort(key=lambda tup: tup[1])
    # invert so the highest score is first:
    # pdb.set_trace()
    tupes.reverse()
    block = len(tupes) // 10
    rec_denom = all_pos(tupes)

    for num in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        upto = num * block
        upto += 1
        precision = prec(tupes[0:upto])
        recall = rec(tupes[0:upto], rec_denom)
        try:
            f1_score = f1(precision, recall)
        except:
            if precision == 0.0 or recall == 0.0:
                f1_score = 0.0
            else:
                pdb.set_trace()
        scores = [x[1] for x in tupes[0:upto]]
        annos = [int(x[0]) for x in tupes[0:upto]]
        aveP = average_precision_score(annos, scores)
        aps = []
        for label in label_names:
            map_tupes = []
            for anno, score, item_label in zip(temp_labels, met[0], labels):
                if item_label == label:
                    map_tupes.append((anno, score))
            ap = average_precision_score([int(x[0]) for x in map_tupes], [x[1] for x in map_tupes])
            if np.isnan(ap):
                ap = 0.0
            aps.append(ap)
        maveP = sum(aps) / len(aps)
        # othermap = aveP / len(annos)
        # meanAvg = get_map(annos)
        print('\t'.join([met_label, str(num * 10), str(precision), str(recall), str(f1_score), str(aveP), str(maveP)]))
    # final block
    precision = prec(tupes)
    recall = rec(tupes, rec_denom)
    f1_score = f1(precision, recall)
    scores = [x[1] for x in tupes]
    annos = [int(x[0]) for x in tupes]
    aveP = average_precision_score(annos, scores)
    # repeated from above
    aps = []
    for label in label_names:
        map_tupes = []
        for anno, score, item_label in zip(temp_labels, met[0], labels):
            if item_label == label:
                map_tupes.append((anno, score))
        ap = average_precision_score([int(x[0]) for x in map_tupes], [x[1] for x in map_tupes])
        if np.isnan(ap):
            ap = 0.0
        aps.append(ap)
    maveP = sum(aps) / len(aps)
    print('\t'.join([met_label, str(100), str(precision), str(recall), str(f1_score), str(aveP), str(maveP)]))



