
# coding: utf-8

# In[2]:


import subprocess
import pdb


# In[3]:


import os; os.getcwd()


# In[46]:


feats = open('genpara.maxent.feats', 'r').readlines()


# In[51]:


# temp_scores[1]


# In[49]:


# glove_src_para_sim


# In[85]:


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
glove_src_para_david = [float(x[2]) for x in temp_scores]
glove_src_orig_sim = [float(x[3]) for x in temp_scores]
glove_src_orig_dist = [float(x[4]) for x in temp_scores]
glove_src_orig_david = [float(x[5]) for x in temp_scores]
glove_orig_para_sim = [float(x[6]) for x in temp_scores]
glove_orig_para_dist = [float(x[7]) for x in temp_scores]
glove_orig_para_david = [float(x[8]) for x in temp_scores]
glove_align_para_sim = [float(x[9]) for x in temp_scores]
glove_align_para_dist = [float(x[10]) for x in temp_scores]
glove_align_para_david = [float(x[11]) for x in temp_scores]
w2v_src_para_sim = [float(x[12]) for x in temp_scores]
w2v_src_para_dist = [float(x[13]) for x in temp_scores]
w2v_src_para_david = [float(x[14]) for x in temp_scores]
w2v_src_orig_sim = [float(x[15]) for x in temp_scores]
w2v_src_orig_dist = [float(x[16]) for x in temp_scores]
w2v_src_orig_david = [float(x[17]) for x in temp_scores]
w2v_orig_para_sim = [float(x[18]) for x in temp_scores]
w2v_orig_para_dist = [float(x[19]) for x in temp_scores]
w2v_orig_para_david = [float(x[20]) for x in temp_scores]
w2v_align_para_sim = [float(x[21]) for x in temp_scores]
w2v_align_para_dist = [float(x[22]) for x in temp_scores]
w2v_align_para_david = [float(x[23]) for x in temp_scores]
elmo_src_para_sim = [float(x[24]) for x in temp_scores]
elmo_src_para_dist = [float(x[25]) for x in temp_scores]
elmo_src_para_david = [float(x[26]) for x in temp_scores]
elmo_src_orig_sim = [float(x[27]) for x in temp_scores]
elmo_src_orig_dist = [float(x[28]) for x in temp_scores]
elmo_src_orig_david = [float(x[29]) for x in temp_scores]
elmo_orig_para_sim = [float(x[30]) for x in temp_scores]
elmo_orig_para_dist = [float(x[31]) for x in temp_scores]
elmo_orig_para_david = [float(x[32]) for x in temp_scores]
elmo_align_para_sim = [float(x[33]) for x in temp_scores]
elmo_align_para_dist = [float(x[34]) for x in temp_scores]
elmo_align_para_david = [float(x[35]) for x in temp_scores]


metrics = [(normalize(glove_src_para_sim), 'glove_src_para_sim'),
    (normalize_and_invert(glove_src_para_dist), 'glove_src_para_dist'),
    (normalize_and_invert(glove_src_para_david), 'glove_src_para_david'),
    (normalize(glove_src_orig_sim), 'glove_src_orig_sim'),
    (normalize_and_invert(glove_src_orig_dist), 'glove_src_orig_dist'),
    (normalize_and_invert(glove_src_orig_david), 'glove_src_orig_david'),
    (normalize(glove_orig_para_sim), 'glove_orig_para_sim'),
    (normalize_and_invert(glove_orig_para_dist), 'glove_orig_para_dist'),
    (normalize_and_invert(glove_orig_para_david), 'glove_orig_para_david'),
    (normalize(glove_align_para_sim), 'glove_align_para_sim'),
    (normalize_and_invert(glove_align_para_dist), 'glove_align_para_dist'),
    (normalize_and_invert(glove_align_para_david), 'glove_align_para_david'),
    (normalize(w2v_src_para_sim), 'w2v_src_para_sim'),
    (normalize_and_invert(w2v_src_para_dist), 'w2v_src_para_dist'),
    (normalize_and_invert(w2v_src_para_david), 'w2v_src_para_david'),
    (normalize(w2v_src_orig_sim), 'w2v_src_orig_sim'),
    (normalize_and_invert(w2v_src_orig_dist), 'w2v_src_orig_dist'),
    (normalize_and_invert(w2v_src_orig_david), 'w2v_src_orig_david'),
    (normalize(w2v_orig_para_sim), 'w2v_orig_para_sim'),
    (normalize_and_invert(w2v_orig_para_dist), 'w2v_orig_para_dist'),
    (normalize_and_invert(w2v_orig_para_david), 'w2v_orig_para_david'),
    (normalize(w2v_align_para_sim), 'w2v_align_para_sim'),
    (normalize_and_invert(w2v_align_para_dist), 'w2v_align_para_dist'),
    (normalize_and_invert(w2v_align_para_david), 'w2v_align_para_david'),
    (normalize(elmo_src_para_sim), 'elmo_src_para_sim'),
    (normalize_and_invert(elmo_src_para_dist), 'elmo_src_para_dist'),
    (normalize_and_invert(elmo_src_para_david), 'elmo_src_para_david'),
    (normalize(elmo_src_orig_sim), 'elmo_src_orig_sim'),
    (normalize_and_invert(elmo_src_orig_dist), 'elmo_src_orig_dist'),
    (normalize_and_invert(elmo_src_orig_david), 'elmo_src_orig_david'),
    (normalize(elmo_orig_para_sim), 'elmo_orig_para_sim'),
    (normalize_and_invert(elmo_orig_para_dist), 'elmo_orig_para_dist'),
    (normalize_and_invert(elmo_orig_para_david), 'elmo_orig_para_david'),
    (normalize(elmo_align_para_sim), 'elmo_align_para_sim'),
    (normalize_and_invert(elmo_align_para_dist), 'elmo_align_para_dist'),
    (normalize_and_invert(elmo_align_para_david), 'elmo_align_para_david')]

new_feats = []
for annot in temp_labels:
    annot += '\t'
    new_f = []
    for met in metrics:
        new_f.append(met[0].pop(0))
    new_f = [str(x) for x in new_f]
    annot += ' '.join(new_f)
    new_feats.append(annot)
    
    
feats = new_feats
    
        


# In[84]:


new_feats[-5]


# In[83]:


feats[-5]


# In[86]:


labels = ['glove_src_para_sim',
            'glove_src_para_dist',
            'glove_src_para_david',
            'glove_src_orig_sim',
            'glove_src_orig_dist',
            'glove_src_orig_david',
            'glove_orig_para_sim',
            'glove_orig_para_dist',
            'glove_orig_para_david',
            'glove_align_para_sim',
            'glove_align_para_dist',
            'glove_align_para_david',
            'w2v_src_para_sim',
            'w2v_src_para_dist',
            'w2v_src_para_david',
            'w2v_src_orig_sim',
            'w2v_src_orig_dist',
            'w2v_src_orig_david',
            'w2v_orig_para_sim',
            'w2v_orig_para_dist',
            'w2v_orig_para_david',
            'w2v_align_para_sim',
            'w2v_align_para_dist',
            'w2v_align_para_david',
            'elmo_src_para_sim', 
            'elmo_src_para_dist', 
            'elmo_src_para_david',
            'elmo_src_orig_sim', 
            'elmo_src_orig_dist', 
            'elmo_src_orig_david',
            'elmo_orig_para_sim', 
            'elmo_orig_para_dist', 
            'elmo_orig_para_david',
            'elmo_align_para_sim', 
            'elmo_align_para_dist', 
            'elmo_align_para_david',
(normalize(ng_src_para), 'ng_src_para'),
(normalize(ng_src_orig), 'ng_src_orig'),
(normalize(ng_orig_para), 'ng_orig_para'),
(normalize(ng_align_para), 'ng_align_para')]


# In[9]:


# test
# len(labels)
# len(feats[0].split('\t')[1].split(' '))
# assert(len(labels) == len(feats[0].split('\t')[1].split(' ')))


# In[87]:


fold_size = len(feats) // 10
remainder = len(feats) % 10
folds = []
for i in range(10):
    start = i * fold_size
    end = (i+1) * fold_size
    folds.append(feats[start:end])
for remaining in feats[-remainder:]:
    folds[-1].append(remaining)


# In[88]:


split_folds = []
for i in range(10):
    dev = folds[i]
    test = folds[(i+1) % 10] + folds[(i+2) % 10]
    train = []
    for j in range(10)[3:]:
        j += i
        j %= 10
        train += folds[j]
    split_folds.append((train, dev, test))
    


# In[89]:


# def maxent_ablation(labels, feats):
start = 0
annotations = []
# scores = []
for fold in range(10):
    k_th_fold = []
    for k in range(3):
        fold_annotations = []
        fold_scores = []
        for sets in split_folds[fold][k]:
            try:
                fold_annotations.append(sets.split('\t')[0])
                fold_scores.append(sets.split('\t')[1])
            except:
                print('repaired set num', start)
                print('set', sets)
                fold_annotations.append(sets.split(' ')[0])
                fold_scores.append(sets.split(' ')[1])
            start += 1
        k_th_fold.append((fold_annotations, fold_scores))
    annotations.append(k_th_fold)
    k_th_fold = []
    


# In[111]:


record = {}
for l in labels:
    record[l] = {}
    
# cmd = "megam_i686.opt -pa -tune perceptron "
cmd = "megam_i686.opt -tune binomial temp.feats > temp.cls"

# test = subprocess.Popen(cmd, shell=True,
#                            stdout=subprocess.PIPE, 
#                            stderr=subprocess.PIPE)
# out, err = test.communicate()


# In[124]:


def write_line(scores, annotations, outfile):
    # print("len annos", len(annotations))
    # print("len scores", len(scores))
    start = 0
    for s, a in zip(scores, annotations):
        # print('on', start, 'of', len(annotations), 'with', a + '\t' + s)
        # we need some cleaning
        outfile.write(a + '\t' + s)
        start += 1

def rm_feat(scores, idx):
    new_scores = []
    for s in scores:
        s = s.strip().split(' ')
        # pull 1 out vs:
        # new_s = s[0:idx] + s[idx + 1:]
        # error_msg = "len s " + str(len(s)) + " len new s " + str(len(new_s))
        # assert(len(s) - 1 == len(new_s)),error_msg
        # only use one
        new_s = [s[idx]]
        # pdb.set_trace()
        # error_msg = "len s " + str(len(s)) + " len new s " + str(len(new_s))
        # assert(len(new_s) == 1),error_msg
        new_s = ' '.join(new_s)
        new_s += '\n'
        new_scores.append(new_s)
    return new_scores

def get_err(err_line):
    pass


# In[125]:


for label in labels:
    idx = labels.index(label)
    foldnum = 0
    for fold in annotations:
        # print('doing', label, foldnum)
        train_score = rm_feat(fold[0][1], idx)
        train_annotation = fold[0][0]
        dev_score = rm_feat(fold[1][1], idx)
        dev_annotation = fold[1][0]
        test_score = rm_feat(fold[2][1], idx)
        test_annotation = fold[2][0]
        outfile = open('temp.feats', 'w')
        # print("doing train")
        write_line(train_score, train_annotation, outfile)
        outfile.write("DEV\n")
        # print("doing dev")
        write_line(dev_score, dev_annotation, outfile)
        outfile.write("TEST\n")
        # print("doing test")
        write_line(test_score, test_annotation, outfile)
        outfile.close()
        out_err = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = out_err.communicate()
        err = err.decode('utf-8')
        err = err.split('\n')
        print(label, foldnum)
        err_list = err[-2].split(' ')
        get_idx = lambda x, y: y.index(x) + 1
        # pdb.set_trace()
        train_err = err_list[get_idx('er', err_list)]
        dev_err = err_list[get_idx('der', err_list)]
        test_err = err_list[get_idx('ter', err_list)]
        # print(err[-2])
        # try:
        assert(foldnum not in record[label])
        # except:
        # pdb.set_trace()

        record[label][foldnum] = (train_err, dev_err, test_err)
        foldnum += 1
        
    


# In[126]:


# record


# In[95]:


import numpy as np

def get_avgs(recorded):
    accs = {}
    for label in recorded:
        train_acc = []
        dev_acc = []
        test_acc = []
        for fold in recorded[label]:
            train_acc.append(float(recorded[label][fold][0]))
            dev_acc.append(float(recorded[label][fold][1]))
            test_acc.append(float(recorded[label][fold][2]))
        print('train_acc', train_acc)
        print('dev_acc', dev_acc)
        print('test_acc', test_acc)
        print(np.mean(train_acc), np.mean(dev_acc), np.mean(test_acc))
        accs[label] = (np.mean(train_acc), np.mean(dev_acc), np.mean(test_acc))
    return accs

accs = get_avgs(record)


# In[96]:


# import ggplot2

out_tsv = open('ten_fold_results.tsv', 'w')
header = "missing\ttrain error\tdev error\ttest error\n"
out_tsv.write(header)
for label in accs:
    out_tsv.write('\t'.join([label, str(accs[label][0]), str(accs[label][1]), str(accs[label][2])]) + '\n')
out_tsv.close()


# In[97]:


accs

