# makeAutTemplates5.py by Sarah Ewing last updated 4/6/2019
# this is meant to do the same thing as makeAutTemplates4.py
# except using ELMo alignments instead of Gold alignments

import pickle, csv, nltk


stopwords = {'is', 'are', 'what', 'who', 'where', 'when', 'how', 'why', 'you', 'your', 'have', 'has', 'be', 'to', 'do'}
punct = {'\'', '.', ',', '!', '?', ';', '\\', '/', ':'}

templates = []
with open('fullAlign.tsv', 'r') as aligns:
    read = csv.reader(aligns, delimiter='\t')
    for i, line in enumerate(read):
        aligns, aligns2, finalaligns = [], [], []
        sent1a = line[0].split()
        for j in range(len(sent1a)-1, -1, -1):
            if '\"' in sent1a[j]:
                sent1a = None
                break
            else:
                for item in punct:
                    if item in sent1a[j]:
                        sent1a[j] = sent1a[j].split(item)[0]
                        sent1a.insert(j+1, item)
        if sent1a is not None:
            try:
                sent1b = nltk.pos_tag(sent1a)
            except IndexError:
                sent1a = None
        sent2a = line[1].split()
        for j in range(len(sent2a)-1, -1, -1):
            if '\"' in sent2a[j]:
                sent2a = None
                break
            else:
                for item in punct:
                    if item in sent2a[j]:
                        sent2a[j] = sent2a[j].split(item)[0]
                        sent2a.insert(j+1, item)
        if sent2a is not None:
            try:
                sent2b = nltk.pos_tag(sent2a)
            except IndexError:
                sent2a = None
        align = line[2].split()
        variable = 0
        if sent1a is not None and sent2a is not None:
            if not align == ['NoAlignments']:
                for item in align:
                    x, y = item.split('-')
                    aligns.append(([int(x)], [int(y)]))
                for pair in align:
                    x, y = pair.split('-')
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
                for (left, right) in aligns2:
                    l, r = 0, 0
                    try:
                        for i in left:
                            if not sent1a[i] in stopwords: l += 1
                        for i in right:
                            if not sent2a[i] in stopwords: r += 1
                        if l > 0 or r > 0:
                            finalaligns.append((left, right))
                    except IndexError:
                        pass
                for i in range(0, len(finalaligns)):
                    tpl = finalaligns[i]
                    left, right = tpl[0], tpl[1]
                    if len(left) == 1 and len(right) == 1:
                        ## ADD POS TAG
                        sent1a[left[0]] = "$" + str(variable) + '/' + sent1b[left[0]][1]
                        sent2a[right[0]] = "$" + str(variable) + '/' + sent2b[right[0]][1]
                        variable += 1
                    else:
                        j, k = len(left)-1, len(right)-1
                        tag_left = []
                        tag_right = []
                        while j >= 0:
                            if j == 0:
                                tag_left.insert(0, sent1b[left[j]][1])
                                sent1a[left[j]] = "$" + str(variable) + '//' + str(tag_left)
                                j -= 1
                            else:
                                tag_left.insert(0, sent1b[left[j]][1])
                                sent1a[left[j]] = "REPLACE"
                                j -= 1
                        while k >= 0:
                            if k == 0:
                                tag_right.insert(0, sent2b[right[k]][1])
                                sent2a[right[k]] = "$" + str(variable) + '//' + str(tag_right)
                                k -= 1
                            else:
                                sent2a[right[k]] = "REPLACE"
                                tag_right.insert(0, sent2b[right[k]][1])
                                k -= 1
                        variable += 1
                while "REPLACE" in sent1a: sent1a.remove("REPLACE")
                while "REPLACE" in sent2a: sent2a.remove("REPLACE")
                i, j = 0, 0
                for word in sent1a:
                    if "$" not in word:
                        i += 1
                if i != 0 and i != len(sent1a):
                    for word in sent2a:
                        if "$" not in word:
                            j += 1
                    if j != 0 and j != len(sent2a):
                        if not sent1a == sent2a:
                            templates.append((sent1a, sent2a))
                            templates.append((sent2a, sent1a))


pickle.dump(templates, open('autTemplates5.p', 'wb'))
print(len(templates))

