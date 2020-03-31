import pickle

templates = pickle.load(open('autTemplateswithSources.p', 'rb'))

used = []
for (sent1, sent2, source1, source2) in templates:
    inds1 = []
    for word in sent1:
        if '$' in word:
            word = word.split('/')
            inds1.append(int(word[0][1:]))
    for i in range(1, len(inds1)-1):
        if inds1[i] == inds1[i-1]+1 and inds1[i+1] == inds1[i]+1:
            if not (sent1, sent2, source1, source2) in used:
                used.append((sent1, sent2, source1, source2))
        else:
            for word in sent1:
                if '$' in word:
                    if 'A' not in word:
                        place = sent1.index(word)
                        splitWord = word.split('/')
                        ind = splitWord[0][1:]
                        new = word[0]+'A'+ind+word[2:]
                        sent1.remove(word)
                        sent1.insert(place, new)
                        for word2 in sent2:
                            if '$' in word2:
                                splitWord2 = word2.split('/')
                                if word2[0][1:] == ind:
                                    place2 = sent2.index(word2)
                                    new2 = word2[0]+'A'+ind+word2[2:]
                                    sent2.remove(word2)
                                    sent2.insert(place2, new2)
            var = 0
            for f in range(0, len(sent1)):
                if '$' in sent1[f]:
                    word = sent1[f].split('/')
                    ind = word[0][1:]
                    if len(word) > 1:
                        sent1[f] = word[0][0] + str(var) + '/' + word[1]
                    else:
                        sent1[f] = word[0][0] + str(var)
                    for l in range(0, len(sent2)):
                        if '$' in sent2[l]:
                            word2 = sent2[l].split('/')
                            if word2[0][1:] == ind:
                                if len(word2) > 1:
                                    sent2[l] = word2[0][0] + str(var) + '/' + word2[1]
                                else:
                                    sent2[l] = word2[0][0] + str(var)
                    var += 1
            fail = 0
            for word in sent2:
                if '$' in word:
                    ind = word.split('/')[0][1:]
                    if int(ind) >= var:
                        fail += 0
            if not (sent1, sent2, source1, source2) in used and fail == 0:
                used.append((sent1, sent2, source1, source2))

print(len(used))
pickle.dump(used, open('autTemplatesCheckedwithSources.p', 'wb'))
