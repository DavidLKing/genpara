# makePar3.py by Sarah Ewing
# This is an attempt to make my work readable
# The file below reads a pickle file called 'bestmatches3.p' and makes automatic
# paraphrases from the information stored in it 'bestmatches3.p' is formatted in the
# following way: it is a list in which each entry is a 4-tuple, the file
# 'wilkins.csv' contains all data needed to construct 'bestmatches3.p', the first
# member of the tuple is the gold label given to the matched observed sentence
# in 'wilkins.csv', the second is that matched sentence, the third is the
# so-called 'left' side of the automatically generated template which has matched
# on the specific wilkins sentence, and the last member of each tuple is the
# 'right' side of the tuple which is a template of how the wilkins sentence should
# be transformed to create the new paraphrase.


import pickle
import csv
import nltk

wnl = nltk.stem.WordNetLemmatizer()


def extract_variables(par, matched, wilks, trans, j, p, called, pulled, chan):
    # feed in the matched template, and everything after 'beg' in wilkins
    wilkstag = nltk.pos_tag(wilks)
    for i in range(0, len(matched)):
        if '$' not in matched[i]:
            k = wilks[p:].index(matched[i])+p
            if j < k:
                if '/' not in matched[i-1]:
                    for m in range(j, k):
                        if not par == []:
                            par[-1] += ' '+wilks[m]
                        else:
                            par.append(wilks[m])
                else:
                    return None, j, called, pulled, chan
                j = k+1
            elif j == k:
                j += 1
            p = k
        elif '/' in matched[i]:
            ind = 0
            b = i-1
            f = i+1
            bef = 1
            single = 0
            tag = matched[i].split('/')[1]
            var = matched[i].split('/')[0]
            while b >- 1:
                if '$' not in matched[b] or ('$' in matched[b] and '/' in matched[b]):
                    break
                else:
                    bef += 1
                    b -= 1
            while f < len(matched):
                if '$' not in matched[f]:
                    break
                elif '/' in matched[f]:
                    single += 1
                    f += 1
                else:
                    f += 1
            if f-single == i+1:
                if f == len(matched):
                    word = wilks[-(single+1)]
                    ind = -(single+1)
                    for q in range(j, len(wilks)+ind):
                        par[-1] += ' '+wilks[q]
                else:
                    word = wilks[wilks.index(matched[f])-(single+1)]
                    ind = wilks.index(matched[f])-(single+1)
                    for q in range(j, ind):
                        par[-1]+=' '+wilks[q]
                    j = ind
            else:
                if not b == -1:
                    if '/' in matched[b]:
                        beg = wilks.index(par[-1])+bef
                    else:
                        beg = wilks.index(matched[b])+bef
                else:
                    beg = bef
                if not f == len(matched):
                    end = wilks[f:].index(matched[f])+f
                else:
                    end = len(wilks)
                for x in range(beg, end):
                    if wilkstag[x][1] == tag:
                        word = wilkstag[x][0]
                        ind = x
                        break
                    else:
                        if not par == []:
                            par[-1] += ' '+wilks[x]
                            j += 1
                        else:
                            par.append(wilks[x])
                            j += 1
            for item in trans:
                if var in item:
                    ttag = item.split('/')[1]

            if '/' in matched[i - 1] and j < ind - 1:
                return None, j, called, pulled, chan

            if '$' in matched[i-1]:
                for w in range(j, ind):
                    if par == []:
                        par.append(wilks[w])
                    else:
                        par[-1] += ' ' + wilks[w]

            try:
                newWord, called, pulled, chan = change_pos(word, tag, ttag, called, pulled, chan)
                par.append(newWord)
            except UnboundLocalError:
                return None, j, called, changed, extract

            if ind >= 0:
                j = ind + 1
            else:
                j = len(wilks)
        else:
            par.append(wilks[j])
            j += 1
    return par, j, called, pulled, chan


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return 'a'
    elif treebank_tag.startswith('V'):
        return 'a'
    elif treebank_tag.startswith('N'):
        return 'n'
    elif treebank_tag.startswith('R'):
        return 'r'
    else:
        return None


lookup = pickle.load(open('lookup.pkl', 'rb'))


def change_pos(word, pos1, pos2, called, pulled, chan):
    if pos1 != pos2:
        chan += 1
    pos = get_wordnet_pos(pos1)
    if pos is None:
        lemma = wnl.lemmatize(word)
    else:
        lemma = wnl.lemmatize(word, pos)
    called += 1
    try:
        new = lookup[lemma][pos2]
        pulled += 1
        # if new != word:
        #     chan += 1
    except KeyError:
        new = word
    return new, called, pulled, chan


def make_par(wilks, sent, trans, paraphrase, par, i):
    for item in trans:
        if '$' not in item:
            paraphrase.append(item)
        else:
            l = int(item[1])
            try:
                paraphrase.append(par[l+i])
            except IndexError:
                return None
    return paraphrase


matches = pickle.load(open('bestmatches.p', 'rb'))
# matches = [('is there anything else i can help you with',
#             ['ok,', 'so', 'we', 'will', 'address', 'your', 'back', 'pain', 'then', 'today', 'in', 'our', 'time', 'allotted.', 'is', 'there', 'anything', 'else', 'you', 'wanted', 'to', 'address'],
#             ['$0', '$1/NN', 'else', 'you', '$2'], ['$0', '$1/NNS', '$2', 'you']),
#            ('are you having any other pain', ['have', 'you', 'noticed', 'the', 'pain', 'radiating', 'to', 'other', 'parts', 'of', 'your', 'body'],
#             ['have', 'you', 'noticed', '$1/DT', '$2/NN', '$3', '$0'],['besides', 'having', 'to', '$0', 'more', 'frequently', ',', 'have', 'you', 'had', '$1/DT', 'other', '$2/NNS', '$3'])]
print(len(matches))
used, used2 = [], []
called, changed, breaks, paraphrases, extract = 0, 0, 0, 0, 0
secCall = 0
with open('autParaphrasesforAnnotation.csv', 'w') as write:
    writer = csv.writer(write)
    writer.writerow(['PARAPHRASE', 'ORIGINAL', 'TEMP MATCH', 'TEMP TRANSFORMATION', 'ORIGINAL LABEL'])
    for (lab, wilks, sent, trans) in matches:
        secCall += 1
        j, f, beg, par = 0, 0, [], []
        if '$' not in sent[0]:
            j = wilks.index(sent[0])+1
            for i in range(0, wilks.index(sent[0])):
                beg.append(wilks[i])
        beg = ' '.join(beg)
        if not beg == '':
            par.append(beg)
        par, j, called, changed, extract = extract_variables(par, sent, wilks, trans, j, f, called, changed, extract)
        if par is None:
            breaks += 1
        else:
            if not j == -1:
                if j < len(wilks):
                    if '$' in sent[-1] and '/' not in sent[-1]:
                        for i in range(j, len(wilks)):
                            try:
                                par[-1]=par[-1]+' '+wilks[i]
                            except IndexError:
                                print('Index Error2')
                                print('i,j,k', i, j, k)
                                print('par', par)
                                print('wilks', wilks)
                    else:
                        add = []
                        for i in range(j, len(wilks)):
                            add.append(wilks[i])
                        add = ' '.join(add)
                        if not add == '':
                            par.append(add)

# par contains the information from wilks stored within each variable
# below the paraphrase is created from this information

            v, paraphrase, i = 0, [], 0
            for item in trans:
                if '$' in item:
                    v += 1
            if v == len(par):
                paraphrase = make_par(wilks, sent, trans, paraphrase, par, i)
            elif (v+1) == len(par):
                if '$' in sent[0]:
                    paraphrase = make_par(wilks, sent, trans, paraphrase, par, i)
                    if paraphrase is not None:
                        paraphrase.append(par[-1])
                else:
                    paraphrase.append(par[0])
                    i += 1
                    paraphrase = make_par(wilks, sent, trans, paraphrase, par, i)
            elif (v+2) == len(par):
                paraphrase.append(par[0])
                i = 1
                paraphrase = make_par(wilks, sent, trans, paraphrase, par, i)
                if paraphrase is not None:
                    paraphrase.append(par[-1])
            else:
                print('ERROR')
                print('V', v)
                print('par', len(par), par)

            if paraphrase is not None:
                paraphrase = ' '.join(paraphrase)
                paraphrases += 1

# print('# of paraphrases', paraphrases)
# print('Called', called)
# print('Pulled', changed)
# print('Changed', extract)

        # print('wilks', wilks)
        # print('matched', sent)
        # print('trans', trans)
        # print('par', par)
        # print('paraphrase', paraphrase)
        # print()

# now we write out to the file 'autParaphrasesforAnnotation.csv'
        if paraphrase is not None:
            row = [paraphrase, wilks, sent, trans, lab]
            writer.writerow(row)
