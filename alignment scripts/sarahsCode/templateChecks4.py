## templateChecks4.py last updated by Sarah Ewing on 2/22/19
## this files takes a pickle file made by 'makeAutTemplates4.py' and fixes any errors in indices of the templates
## contained in it
## this forces the variables in the left side of the template to be in ascending order starting from zero
## this, as with all version 4 files has both 1-1 and non 1-1 variables with specified POS tags
## the next file in this series is 'makeMatches4.py'

import pickle

templates = pickle.load(open('autTemplates4.p', 'rb'))

print(len(templates))
used = []


def alter_variables(sent1, sent2, failed):
    for word in sent1:
        if '$' in word:
            place = sent1.index(word)
            splitWord = word.split('//')
            if len(splitWord) == 1:
                splitWord = word.split('/')
            ind = splitWord[0][1:]
            if 'A' not in ind:
                new = word[0] + 'A' + ind + word[2:]
                sent1.remove(word)
                sent1.insert(place, new)
    for word2 in sent2:
        if '$' in word2:
            place = sent2.index(word2)
            splitWord = word2.split('//')
            if len(splitWord) == 1:
                splitWord = word2.split('/')
            ind = splitWord[0][1:]
            if 'A' not in ind:
                new = word2[0] + 'A' + ind + word2[2:]
                sent2.remove(word2)
                sent2.insert(place, new)
    var = 0
    for f in range(0, len(sent1)):
        if '$' in sent1[f]:
            double = True
            word = sent1[f].split('//')
            if len(word) == 1:
                word = sent1[f].split('/')
                double = False
            ind = word[0][1:]
            if double is True:
                sent1[f] = word[0][0] + str(var) + '//' + word[1]
            else:
                sent1[f] = word[0][0] + str(var) + '/' + word[1]
            for l in range(0, len(sent2)):
                if '$' in sent2[l]:
                    word2 = sent2[l].split('//')
                    if len(word2) == 1:
                        word2 = sent2[l].split('/')
                    if word2[0][1:] == ind:
                        if double == True:
                            sent2[l] = word2[0][0] + str(var) + '//' + word2[1]
                        else:
                            sent2[l] = word2[0][0] + str(var) + '/' + word2[1]

            var += 1
    fail = 0
    for word in sent2:
        if '$' in word:
            ind = word.split('/')[0][1:]
            try:
                if int(ind) >= var:
                    fail += 0
            except ValueError:
                print('sent1', sent1)
                print('sent2', sent2)
    if not (sent1, sent2, source1, source2) in used and fail == 0:
        used.append((sent1, sent2, source1, source2))
    else:
        failed += 1



fails = 0
for (sent1, sent2, source1, source2) in templates:
    left = 0
    right = 0
    for word in sent1:
        if '$' in word:
            left += 1
    for word in sent2:
        if '$' in word:
            right += 1
    if left != right:
        break
    inds1 = []
    success = 0
    for word in sent1:
        if '$' in word:
            word = word.split('/')
            inds1.append(int(word[0][1:]))
    if len(inds1) > 2:
        for i in range(1, len(inds1)-1):
            if inds1[i] == inds1[i-1]+1 and inds1[i+1] == inds1[i]+1:
                success += 1
            if success == len(inds1)-2:
                if not (sent1, sent2, source1, source2) in used:
                    used.append((sent1, sent2, source1, source2))
                else:
                    fails += 1
            else:
                alter_variables(sent1, sent2, fails)
    elif inds1 == [0] or inds1 == [0, 1]:
        if not (sent1, sent2, source1, source2) in used:
            used.append((sent1, sent2, source1, source2))
        else:
            fails += 1
    else:
        alter_variables(sent1, sent2, fails)


for (sent1, sent2, source1, source2) in templates:
    if not (sent1, sent2, source1, source2) in used:
        i = 0
        f = False
        for word in sent1:
            if '$' in word:
                if int(word[1]) == i:
                    i += 1
                else:
                    f = True
        if f is False:
            if not (sent1, sent2) in used:
                used.append((sent1, sent2, source1, source2))

print(len(used))
print(fails)
pickle.dump(used, open('autTemplatesChecked4.p', 'wb'))
