# makeMatches3.py last updated by Sarah Ewing 11/28/18
# Matches sentences from Wilkins to created templates from
# makeAutTemplates3.py and refined by templateChecks.py
f


import pickle, csv, difflib

matches = []
wilks = []


def doDiff(lab, trans, sent, wilks):
    diff = difflib.ndiff(sent, wilks)
    pos, neg, w, c, tag, count = 0, 0, 0, 0, 0, 0
    for item in sent:
        if not '$' in item:
            w += 1
    dif = []
    for item in diff:
        dif.append(item)
    for i in range(0, len(dif) + 1):
        if i == len(dif):
            if (neg > tag and c == w and neg <= pos) or (neg == tag and neg == pos and c == w):
                return (lab, wilks, sent, trans)
        elif dif[i][0] == ' ':
            c += 1
            if count != 0:
                break
            elif neg == pos and neg == tag:
                neg, pos, tag = 0, 0, 0
            elif neg <= pos and neg > tag:
                neg, pos, tag = 0, 0, 0
            else:
                break
        elif dif[i][0] == '-':
            neg += 1
            num = dif[i][1:].split('/')[1]
            if type(num) is int:
                count += num
            elif type(num) is str:
                count += 1
        elif dif[i][0] == '+':
            pos += 1
            num = dif[i][1:].split('/')[1]
            if type(num) is int:
                count += num
            elif type(num) is str:
                count += 1


templates = pickle.load(open('templates3.p', 'rb'))
print(len(templates))
with open('wilkins.csv', 'r') as wilkins:
    wilkins = csv.reader(wilkins, delimiter='\t')
    for line in wilkins:
        wilks.append((line[0].split(), line[1].split(), line[3]))

for (sent1, sent2) in templates:
    for (wilk0, wilk1, wilk3) in wilks:
        w, x = doDiff(wilk3, sent2, sent1, wilk0), doDiff(wilk3, sent2, sent1, wilk1)
        if w is not None:
            if w not in matches:
                matches.append(w)
        if x is not None:
            if x not in matches:
                matches.append(x)

print(len(matches))
pickle.dump(matches, open('matches3.p', 'wb'))
