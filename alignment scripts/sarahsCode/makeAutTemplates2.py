## makeAutTemplates2.py last updated by Sarah Ewing 2/14/19
## This file should take the gold alignments from the goldalign-repo-master file
## and make paraphrase templates from them.  Version two will incorporate POS
## tags in one-to-one alignments 
## this version now pulls through the aligned sentences which created the template, and the dial and turn of each of
## them

import pickle, csv, nltk

nums = []
for i in range(0, 191):
    num = ''
    for j in range(0, 4-len(list(str(i)))):
        num += '0'
    num += str(i)
    nums.append(num)
    
stopwords = {'is', 'are', 'what', 'who', 'where', 'when', 'how', 'why', 'you', 'your', 'have', 'has', 'be', 'to', 'them'}

templates = []
templates2 = []
for item in nums:
    filename = 'goldalign-repo-master/data/users/arbit/demo-user-1_demo-user-2/complete/vpd-corpus/batch_'+item+'.tsv'
    try:
        with open(filename, 'r') as file:
            read = csv.reader(file, delimiter='\t')
            for i, line in enumerate(read):
                if line[6] == '1':
                    aligns, aligns2, finalaligns = [],[],[]
                    sent1a = line[1].split()
                    sent1b = nltk.pos_tag(sent1a)
                    sent2a = line[3].split()
                    sent2b = nltk.pos_tag(sent2a)
                    sure = line[7].split()
                    poss = line[8].split()
                    align = sure+poss
                    variable = 0
                    for item in align:
                        x,y = item.split('-')
                        aligns.append(([int(x)],[int(y)]))
                    for pair in align:
                        x,y = pair.split('-')
                        for item in aligns:
                            if int(x) in item[0]:
                                if not int(y) in item[1]:
                                    item[1].append(int(y))
                            elif int(y) in item[1]:
                                item[0].append(int(x))
                    for item in aligns:
                        left = sorted(item[0])
                        right = sorted(item[1])
                        if not (left, right) in aligns2:
                            aligns2.append((left, right))
                    for (left,right) in aligns2:
                        l,r = 0,0
                        for i in left:
                            if not sent1a[i] in stopwords: l+=1
                        for i in right:
                            if not sent2a[i] in stopwords: r+=1
                        if l > 0 or r > 0:
                            finalaligns.append((left,right))
                    for i in range(0, len(finalaligns)):
                        tpl = finalaligns[i]
                        left,right = tpl[0],tpl[1]
                        if len(left)==1 and len(right)==1:
                            ## ADD POS TAG
                            sent1a[left[0]] = "$"+str(variable)+'/'+sent1b[left[0]][1]
                            sent2a[right[0]] = "$"+str(variable)+'/'+sent2b[right[0]][1]
                            variable += 1
                        else:
                            j,k = 0,0
                            while j < len(left):
                                if j==0:
                                    sent1a[left[j]] = "$"+str(variable)
                                    j+=1
                                else:
                                    sent1a[left[j]] = "REPLACE"
                                    j+=1
                            while k < len(right):
                                if k==0:
                                    sent2a[right[k]] = "$"+str(variable)
                                    k+=1
                                else:
                                    sent2a[right[k]] = "REPLACE"
                                    k+=1
                            variable += 1
                    while "REPLACE" in sent1a: sent1a.remove("REPLACE")
                    while "REPLACE" in sent2a: sent2a.remove("REPLACE")
                    i,j = 0,0
                    for word in sent1a:
                        if not "$" in word:
                            i+=1
                    if i!= 0 and i!= len(sent1a):
                        for word in sent2a:
                            if not "$" in word:
                                j+=1
                        if j != 0 and j != len(sent2a):
                            if not sent1a == sent2a:
                                templates.append((sent1a, sent2a))
                                templates.append((sent2a, sent1a))
                                templates2.append((sent1a, sent2a, line[1], line[3]))
                                templates2.append((sent2a, sent1a, line[3], line[1]))
    except FileNotFoundError:
        pass


#pickle.dump(templates, open('autTemplates2.p', 'wb'))
pickle.dump(templates2, open('autTemplateswithSources.p', 'wb'))
print(len(templates))
