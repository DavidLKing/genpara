# makePar4.py by Sarah Ewing last updated 2/26/19
# this file reads in the pickle file called bestMatchesWithSources4.p
# it makes paraphrases based on the information stored in the pickle file
# it writes the file of paraphrases called paraphrasesAllVariablesPOS.csv

import pickle
import csv
import nltk

wnl = nltk.stem.WordNetLemmatizer()
fail = 0

def extract_variables(par, matched, wilks, trans, j, fail):
    # feed in the matched template, and everything after 'beg' in wilkins
    for i in range(0, len(matched)):
        if len(matched[i].split('//')) > 1:
            var = []
            num = len(list(matched[i].split('//')))
            for l in range(j, num):
                var.append(wilks[l])
            var = ' '.join(var)
            j += num
            par.append(var)
        elif len(matched[i].split('/')) > 1:
            try:
                var = wilks[j]
            except IndexError:
                print('wilks', wilks)
                print('j', j)
                print('matched', matched)
                par = None
                fail += 1
                return par, j, fail
            j += 1
            par.append(var)
        else:
            j += 1
    return par, j, fail


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


def change_pos(word, pos1, pos2):
    if pos1 != pos2:
        pos = get_wordnet_pos(pos1)
        if pos is None:
            lemma = wnl.lemmatize(word)
        else:
            lemma = wnl.lemmatize(word, pos)
        try:
            new = lookup[lemma][pos2]
        except KeyError:
            new = word
    else:
        return word
    return new

lookup2 = pickle.load(open('multiLookup.p', 'rb'))

def look_up(variable, trans_list, match_list):
    if trans_list == match_list:
        return variable
    else:
        try:
            var = lookup2[variable][len(trans_list)]
            for item in var:
                return item
        except KeyError:
            return variable



def make_par(trans, paraphrase, par, i, sent):
    for l in range(0, len(trans)):
        if '$' not in trans[l]:
            paraphrase.append(trans[l])
        elif len(trans[l].split('//')) > 1:
            var = trans[l].split('//')[0]
            trans_list = list(trans[l].split('//')[1])
            for item in sent:
                if '$' in item and item.split('/')[0] == var:
                    match_list = list(item.split('//')[1])
            paraphrase.append(look_up(par[i], trans_list, match_list))
            i += 1
        elif len(trans[l].split('/')) > 1:
            var = trans[l].split('/')[0]
            trans_pos = trans[l].split('/')[1]
            for item in sent:
                if '$' in item and item.split('/')[0] == var:
                    match_pos = item.split('/')[1]
            paraphrase.append(change_pos(par[i], match_pos, trans_pos))
            i += 1
    return paraphrase


matches = pickle.load(open('bestMatchesWithSources5.p', 'rb'))

used, used2 = [], []
called, changed, breaks, paraphrases, extract = 0, 0, 0, 0, 0
secCall,check, check2 = 0,0,0
with open('paraphrasesOnePOSManyLength.tsv', 'w') as write:
    writer = csv.writer(write, delimiter='\t')
    writer.writerow(['TEMP MATCH (SWAPPABLE)', 'TEMP TRANSFORMATION (SWAP)', 'SOURCE LEFT (SOURCE ALIGN)',
                     'SOURCE RIGHT (SOURCE ALIGN)', 'PARAPHRASE', 'ORIGINAL',  'LABEL', 'DIALOGUE, TURN'])
    for (lab, wilks, sent, trans, source1, source2, dialTurn) in matches:
        secCall += 1
        j, f, beg, par = 0, 0, [], []
       # if '$' not in sent[0]:
        #    j = wilks.index(sent[0])+1
         #   for i in range(0, wilks.index(sent[0])):
          #      beg.append(wilks[i])
        #beg = ' '.join(beg)
        #if not beg == '':
         #   par.append(beg)
        par, j, fail = extract_variables(par, sent, wilks, trans, j, fail)
        if par is not None:
            check += 1
            if j < len(wilks):
                add = []
                for i in range(j, len(wilks)):
                    add.append(wilks[i])
                add = ' '.join(add)
                if not add == '':
                    par.append(add)

    # par contains the information from wilks stored within each variable
    # below the paraphrase is created from this information
    #             changes = get_changes(par, gold_singles, gold_phrases)
    #            changes, singles, multiples = get_changes(par, elmo_singles, elmo_phrases)
    #            changes = set(changes)
            v, paraphrase, i = 0, [], 0
            for item in trans:
                if '$' in item:
                    v += 1
            if v == len(par):
                paraphrase = make_par(trans, paraphrase, par, i, sent)
            elif (v+1) == len(par):
                check2 += 1
                if '$' in sent[0]:
                    paraphrase = make_par(trans, paraphrase, par, i, sent)
                    paraphrase.append(par[-1])
                else:
                    paraphrase.append(par[0])
                    i += 1
                    paraphrase = make_par(trans, paraphrase, par, i, sent)
            elif (v+2) == len(par):
                paraphrase.append(par[0])
                i = 1
                paraphrase = make_par(trans, paraphrase, par, i, sent)
                paraphrase.append(par[-1])
            else:
                print('ERROR')
                print('V', v)
                print('par', len(par), par)

            paraphrase = ' '.join(paraphrase)
            paraphrases += 1
            row = [sent, trans, source1, source2, paraphrase, wilks, lab, dialTurn]
            writer.writerow(row)
print('fail', fail)
print('secCall', secCall)
print('check', check)
print('check2', check2)
