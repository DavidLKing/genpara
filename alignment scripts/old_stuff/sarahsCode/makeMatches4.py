# makeMatches4.py by Sarah Ewing last updated 2/22/19
# this file takes the templates from 'templateChecks4.py' as a pickle file
# and matches them where appropriate to lines of real dialogue or labels in
# wilkinsWithDialTurn.csv
# do templateChecks4.py before this and refineMatches4.py after

import pickle, csv, difflib

matches = []
wilks = []


def doDiff(lab, trans, sent, wilks, source1, source2, dialTurn):
    diff = difflib.ndiff(sent, wilks)
    pos, neg, w, c, tag = 0, 0, 0, 0, 0
    for item in sent:
        if '$' not in item:
            w += 1
    dif = []
    for item in diff:
        dif.append(item)
    for i in range(0, len(dif) + 1):
        if i == len(dif):
            if (neg > tag and c == w and neg <= pos) or (neg == tag and neg == pos and c == w):
                return (lab, wilks, sent, trans, source1, source2, dialTurn)
        elif dif[i][0] == ' ':
            c += 1
            if neg == pos and neg == tag:
                neg, pos, tag = 0, 0, 0
            elif neg <= pos and neg > tag:
                neg, pos, tag = 0, 0, 0
            else:
                break
        elif dif[i][0] == '-':
            neg += 1
            if '/' in dif[i][2:]:
                tag += 1
        elif dif[i][0] == '+':
            if len(dif[i][2:].split('//')) > 1:
                pos += len(list(dif[i][2:].split('//')[1]))
            else:
                pos += 1


templates = pickle.load(open('autTemplatesChecked4.p', 'rb'))
print(len(templates))

with open('wilkinsWithDialTurn.csv', 'r') as wilkins:
    wilkins = csv.reader(wilkins, delimiter='\t')
    for line in wilkins:
        wilks.append((line[0].split(), line[1].split(), line[3], line[-1]))

for (sent1, sent2, source1, source2) in templates:
    for (wilk0, wilk1, wilk3, dialTurn) in wilks:
        w = doDiff(wilk3, sent2, sent1, wilk0, source1, source2, dialTurn)
        x = doDiff(wilk3, sent2, sent1, wilk1, source1, source2, dialTurn)
        if w is not None:
            if w not in matches:
                matches.append(w)
        if x is not None:
            if x not in matches:
                matches.append(x)

print(len(matches))
pickle.dump(matches, open('matchesWithSources4.p', 'wb'))
