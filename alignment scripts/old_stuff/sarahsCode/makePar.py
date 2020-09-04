## makePar.py by Sarah Ewing
## This is an attempt to make my work readable
## The file below reads a pickle file called 'matches.p' and makes automatic
## paraphrases from the information stored in it 'matches.p' is formatted in the
## following way: it is a list in which each entry is a 4-tuple, the file
## 'wilkins.csv' contains all data needed to construct 'matches.p', the first
## member of the tuple is the gold label given to the matched observed sentence
## in 'wilkins.csv', the second is that matched sentence, the third is the
## so-called 'left' side of the automatically generated template which has matched
## on the specific wilkins sentence, and the last member of each tuple is the
## 'right' side of the tuple which is a teplate of how the wilkins sentence should
## be transformed to create the new paraphrase.
##
##

import pickle
import csv
import nltk

wnl = nltk.stem.WordNetLemmatizer()


def make_par(trans, paraphrase, par, i):
    for item in trans:
        if '$' not in item:
            paraphrase.append(item)
        else:
            paraphrase.append(par[i])
            i += 1
    return paraphrase


def extract_variables(par, left, orig, right, j, p):
    # feed in the matched template, and everything after 'beg' in wilkins
    for i in range(0, len(left)):
        if '$' not in left[i]:
            k = orig[p:].index(left[i])+p
            if j < k:
                if '/' not in left[i-1]:
                    for m in range(j, k):
                        if not par == []:
                            par[-1] += ' '+orig[m]
                        else:
                            par.append(orig[m])
                else:
                    return None, j
                j = k+1
            elif j == k:
                j += 1
            p = k
        # elif '/' in left[i]:
        #     ind = 0
        #     b = i-1
        #     f = i+1
        #     bef = 1
        #     single = 0
        #     tag = left[i].split('/')[1]
        #     var = left[i].split('/')[0]
        #     while b >- 1:
        #         if '$' not in left[b] or ('$' in left[b] and '/' in left[b]):
        #             break
        #         else:
        #             bef += 1
        #             b -= 1
        #     while f < len(left):
        #         if '$' not in left[f]:
        #             break
        #         elif '/' in left[f]:
        #             single += 1
        #             f += 1
        #         else:
        #             f += 1
        #     if f-single == i+1:
        #         if f == len(left):
        #             word = orig[-(single+1)]
        #             ind = -(single+1)
        #             for q in range(j, len(orig)+ind):
        #                 par[-1] += ' '+orig[q]
        #         else:
        #             word = orig[orig.index(left[f])-(single+1)]
        #             ind = orig.index(left[f])-(single+1)
        #             for q in range(j, ind):
        #                 par[-1]+=' '+orig[q]
        #             j = ind
        #     else:
        #         if not b == -1:
        #             if '/' in left[b]:
        #                 beg = orig.index(par[-1])+bef
        #             else:
        #                 beg = orig.index(left[b])+bef
        #         else:
        #             beg = bef
        #         if not f == len(left):
        #             end = orig[f:].index(left[f])+f
        #         else:
        #             end = len(orig)
        #         for x in range(beg, end):
        #             if wilkstag[x][1] == tag:
        #                 word = wilkstag[x][0]
        #                 ind = x
        #                 break
        #             else:
        #                 if not par == []:
        #                     par[-1] += ' '+orig[x]
        #                     j += 1
        #                 else:
        #                     par.append(orig[x])
        #                     j += 1
        #     for item in right:
        #         if var in item:
        #             ttag = item.split('/')[1]
        #
        #     if '/' in left[i - 1] and j < ind - 1:
        #         return None, j
        #
        #     if '$' in left[i-1]:
        #         for w in range(j, ind):
        #             if par == []:
        #                 par.append(orig[w])
        #             else:
        #                 par[-1] += ' ' + orig[w]
        #
        #     try:
        #         newWord, called, pulled, chan = change_pos(word, tag, ttag, called, pulled, chan)
        #         par.append(newWord)
        #     except UnboundLocalError:
        #         return None, j, called, changed, extract
        #
        #     if ind >= 0:
        #         j = ind + 1
        #     else:
        #         j = len(wilks)
        else:
            par.append(orig[j])
            j += 1
    return par, j


matches = pickle.load(open('bestMatchesWithSourcesNoPOS.p', 'rb'))
used, used2 = [], []
with open('autParaphrasesNoPOS.tsv', 'w') as write:
    writer = csv.writer(write, delimiter='\t')
    writer.writerow(['swappable', 'swap', 'src', 'align', 'para', 'orig', 'label', 'response', 'cs_guess', 'cs correct',
                     'color code 1', 'color code 2', 'dialogue, turn'])
    for (label, wilks, sent, trans, source1, source2, response, cs_guess, cs_corr, cc1, cc2, dialTurn) in matches:
        j, f, beg, par = 0, 0, [], []
        if "$" not in sent[0]:
            j = wilks.index(sent[0])+1
            for i in range(0, wilks.index(sent[0])):
                beg.append(wilks[i])
        beg = ' '.join(beg)
        if not beg == '':
            par.append(beg)
        par, j = extract_variables(par, sent, wilks, trans, j, f)
        if not j == -1:
            if j < len(wilks):
                if '$' in sent[-1] and '/' not in sent[-1]:
                    for i in range(j, len(wilks)):
                        try:
                            par[-1] = par[-1] + ' ' + wilks[i]
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
## par contains the information from wilks stored within each variable
## below the paraphrase is created from this information

        v, paraphrase, i = 0, [], 0
        for item in trans:
            if '$' in item:
                v += 1
        if v == len(par):
            paraphrase = make_par(trans, paraphrase, par, i)
        elif (v+1) == len(par):
            if '$' in sent[0]:
                paraphrase = make_par(trans, paraphrase, par, i)
                paraphrase.append(par[-1])
            else:
                paraphrase.append(par[0])
                i += 1
                paraphrase = make_par(trans, paraphrase, par, i)
        elif (v+2) == len(par):
            paraphrase.append(par[0])
            i = 1
            paraphrase = make_par(trans, paraphrase, par, i)
            paraphrase.append(par[-1])
        else:
            print('ERROR')
            print('V', v)
            print('par', len(par), par)

        paraphrase = ' '.join(paraphrase)

# now we write out to the file 'autParaphrasesNoPOS.csv'
        row = [sent, trans, source1, source2, paraphrase, wilks, label, response, cs_guess, cs_corr, cc1, cc2, dialTurn]
        writer.writerow(row)
