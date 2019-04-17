
# coding: utf-8

# In[1]:


import subprocess
import pdb
from sklearn.metrics import average_precision_score

# In[2]:


import os; os.getcwd()


# In[19]:


# feats = open('sanity.feats', 'r').readlines()
feats = open('sanity.nospecial.feats', 'r').readlines()


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

temp_labels = [x.split('\t')[0] for x in feats]

temp_scores = [x.split('\t')[1].split(' ') for x in feats]

[float(x[0]) for x in temp_scores]

glove_src_para_sim = [float(x[0]) for x in temp_scores]
glove_src_para_dist = [float(x[1]) for x in temp_scores]
glove_src_para_joint = [float(x[2]) for x in temp_scores]
glove_src_orig_sim = [float(x[3]) for x in temp_scores]
glove_src_orig_dist = [float(x[4]) for x in temp_scores]
glove_src_orig_joint = [float(x[5]) for x in temp_scores]
glove_orig_para_sim = [float(x[6]) for x in temp_scores]
glove_orig_para_dist = [float(x[7]) for x in temp_scores]
glove_orig_para_joint = [float(x[8]) for x in temp_scores]
glove_align_para_sim = [float(x[9]) for x in temp_scores]
glove_align_para_dist = [float(x[10]) for x in temp_scores]
glove_align_para_joint = [float(x[11]) for x in temp_scores]
w2v_src_para_sim = [float(x[12]) for x in temp_scores]
w2v_src_para_dist = [float(x[13]) for x in temp_scores]
w2v_src_para_joint = [float(x[14]) for x in temp_scores]
w2v_src_orig_sim = [float(x[15]) for x in temp_scores]
w2v_src_orig_dist = [float(x[16]) for x in temp_scores]
w2v_src_orig_joint = [float(x[17]) for x in temp_scores]
w2v_orig_para_sim = [float(x[18]) for x in temp_scores]
w2v_orig_para_dist = [float(x[19]) for x in temp_scores]
w2v_orig_para_joint = [float(x[20]) for x in temp_scores]
w2v_align_para_sim = [float(x[21]) for x in temp_scores]
w2v_align_para_dist = [float(x[22]) for x in temp_scores]
w2v_align_para_joint = [float(x[23]) for x in temp_scores]
elmo_src_para_sim = [float(x[24]) for x in temp_scores]
elmo_src_para_dist = [float(x[25]) for x in temp_scores]
elmo_src_para_joint = [float(x[26]) for x in temp_scores]
elmo_src_orig_sim = [float(x[27]) for x in temp_scores]
elmo_src_orig_dist = [float(x[28]) for x in temp_scores]
elmo_src_orig_joint = [float(x[29]) for x in temp_scores]
elmo_orig_para_sim = [float(x[30]) for x in temp_scores]
elmo_orig_para_dist = [float(x[31]) for x in temp_scores]
elmo_orig_para_joint = [float(x[32]) for x in temp_scores]
elmo_align_para_sim = [float(x[33]) for x in temp_scores]
elmo_align_para_dist = [float(x[34]) for x in temp_scores]
elmo_align_para_joint = [float(x[35]) for x in temp_scores]
ng_src_para = [float(x[36]) for x in temp_scores]
ng_src_orig = [float(x[37]) for x in temp_scores]
ng_orig_para = [float(x[38]) for x in temp_scores]
ng_align_para = [float(x[39]) for x in temp_scores]


metrics = [(normalize(glove_src_para_sim), 'glove_src_para_sim'),
    (normalize_and_invert(glove_src_para_dist), 'glove_src_para_dist'),
    (normalize_and_invert(glove_src_para_joint), 'glove_src_para_joint'),
    (normalize(glove_src_orig_sim), 'glove_src_orig_sim'),
    (normalize_and_invert(glove_src_orig_dist), 'glove_src_orig_dist'),
    (normalize_and_invert(glove_src_orig_joint), 'glove_src_orig_joint'),
    (normalize(glove_orig_para_sim), 'glove_orig_para_sim'),
    (normalize_and_invert(glove_orig_para_dist), 'glove_orig_para_dist'),
    (normalize_and_invert(glove_orig_para_joint), 'glove_orig_para_joint'),
    (normalize(glove_align_para_sim), 'glove_align_para_sim'),
    (normalize_and_invert(glove_align_para_dist), 'glove_align_para_dist'),
    (normalize_and_invert(glove_align_para_joint), 'glove_align_para_joint'),
    (normalize(w2v_src_para_sim), 'w2v_src_para_sim'),
    (normalize_and_invert(w2v_src_para_dist), 'w2v_src_para_dist'),
    (normalize_and_invert(w2v_src_para_joint), 'w2v_src_para_joint'),
    (normalize(w2v_src_orig_sim), 'w2v_src_orig_sim'),
    (normalize_and_invert(w2v_src_orig_dist), 'w2v_src_orig_dist'),
    (normalize_and_invert(w2v_src_orig_joint), 'w2v_src_orig_joint'),
    (normalize(w2v_orig_para_sim), 'w2v_orig_para_sim'),
    (normalize_and_invert(w2v_orig_para_dist), 'w2v_orig_para_dist'),
    (normalize_and_invert(w2v_orig_para_joint), 'w2v_orig_para_joint'),
    (normalize(w2v_align_para_sim), 'w2v_align_para_sim'),
    (normalize_and_invert(w2v_align_para_dist), 'w2v_align_para_dist'),
    (normalize_and_invert(w2v_align_para_joint), 'w2v_align_para_joint'),
    (normalize(elmo_src_para_sim), 'elmo_src_para_sim'),
    (normalize_and_invert(elmo_src_para_dist), 'elmo_src_para_dist'),
    (normalize_and_invert(elmo_src_para_joint), 'elmo_src_para_joint'),
    (normalize(elmo_src_orig_sim), 'elmo_src_orig_sim'),
    (normalize_and_invert(elmo_src_orig_dist), 'elmo_src_orig_dist'),
    (normalize_and_invert(elmo_src_orig_joint), 'elmo_src_orig_joint'),
    (normalize(elmo_orig_para_sim), 'elmo_orig_para_sim'),
    (normalize_and_invert(elmo_orig_para_dist), 'elmo_orig_para_dist'),
    (normalize_and_invert(elmo_orig_para_joint), 'elmo_orig_para_joint'),
    (normalize(elmo_align_para_sim), 'elmo_align_para_sim'),
    (normalize_and_invert(elmo_align_para_dist), 'elmo_align_para_dist'),
    (normalize_and_invert(elmo_align_para_joint), 'elmo_align_para_joint'),
    (normalize_and_invert(ng_src_para), 'ng_src_para'),
    (normalize_and_invert(ng_src_orig), 'ng_src_orig'),
    (normalize_and_invert(ng_orig_para), 'ng_orig_para'),
    (normalize_and_invert(ng_align_para), 'ng_align_para')]


def prec(data):
    total = 0
    num = 0
    for datum in data:
        total += 1
        if datum[0] == '1':
            num += 1
    return num / total

def all_pos(data):
    total = 0
    for datum in data:
        if datum[0] == '1':
            total += 1
    return total

def rec(data, denom):
    num = 0
    for datum in data:
        if datum[0] == '1':
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
        f1_score = f1(precision, recall)
        scores = [x[1] for x in tupes[0:upto]]
        annos = [int(x[0]) for x in tupes[0:upto]]
        # pdb.set_trace()
        aveP = average_precision_score(annos, scores)
        # othermap = aveP / len(annos)
        meanAvg = get_map(annos)
        print('\t'.join([met_label, str(num * 10), str(precision), str(recall), str(f1_score), str(aveP), str(meanAvg)]))
    # final block
    precision = prec(tupes)
    recall = rec(tupes, rec_denom)
    f1_score = f1(precision, recall)
    print('\t'.join([met_label, str(100), str(precision), str(recall), str(f1_score), str(aveP), str(meanAvg)]))



