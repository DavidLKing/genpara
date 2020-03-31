import pickle,csv,difflib

matches = []
wilks = []


def doDiff(lab, trans, sent, wilks, source1, source2, response, cs_guess, cs_corr, cc1, cc2, dialTurn):
    diff = difflib.ndiff(sent, wilks)
    pos, neg, w, c, tag = 0, 0, 0, 0, 0
    for item in sent:
        if '$' not in item:
            w += 1
    dif = []
    for item in diff:
        dif.append(item)
    for i in range(0, len(dif)+1):
        if i == len(dif):
            if (neg > tag and c == w and neg <= pos) or (neg == tag and neg == pos and c == w):
                return lab, wilks, sent, trans, source1, source2, response, cs_guess, cs_corr, cc1, cc2, dialTurn
        elif dif[i][0] == ' ':
            c += 1
            if neg == pos and neg == tag:
                neg, pos, tag = 0, 0, 0
            elif neg <= pos and neg > tag:
                neg, pos, tag = 0, 0, 0
            else:
                break
        elif dif[i][0] == '-':
            neg+=1
            if '/' in dif[i][2:]:
                tag += 1
        elif dif[i][0] == '+':
            pos += 1


templates = pickle.load(open('autTemplatesCheckedwithSources.p', 'rb'))
print(len(templates))

with open('wilkinsWithDialTurn.csv', 'r') as wilkins:
    wilkins = csv.reader(wilkins)
    for line in wilkins:
        wilks.append((line[0].split(), line[1].split(), line[3], line[2], line[1], line[4], line[5], line[6], line[-1]))

for (sent1, sent2, source1, source2) in templates:
    for (wilk0, wilk1, label, response, cs_guess, cs_corr, cc1, cc2, dialTurn) in wilks:
        w = doDiff(label, sent2, sent1, wilk0, source1, source2, response, cs_guess, cs_corr, cc1, cc2, dialTurn)
        x = doDiff(label, sent2, sent1, wilk1, source1, source2, response, cs_guess, cs_corr, cc1, cc2, dialTurn)
        if w is not None:
            if w not in matches:
                matches.append(w)
        if x is not None:
            if x not in matches:
                matches.append(x)
            
print(len(matches))
pickle.dump(matches, open('matchesWithFullSources.p', 'wb'))
