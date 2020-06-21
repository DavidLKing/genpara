import pickle,csv,difflib

matches = []
wilks = []


def doDiff(lab, trans, sent, wilks, source1, source2, response, cs_guess, cs_correct, c_code_1, c_code_2, dialTurn):
    diff = difflib.ndiff(sent,wilks)
    pos, neg, w, c = 0, 0, 0, 0
    for item in sent:
        if '$' not in item:
            w += 1
    dif = []
    for item in diff:
        dif.append(item)
    for i in range(0, len(dif)+1):
        if i == len(dif):
            if neg<=pos and c==w:
                return lab, wilks, sent, trans, source1, source2, response, cs_guess, cs_correct, c_code_1, c_code_2, dialTurn
        elif dif[i][0] == ' ':
            if dif[i-1][0] == ' ':
                if not neg == 0 and not pos == 0:
                    break
            c += 1
            if neg <= pos:
                neg, pos = 0, 0
            else:
                neg, pos = 0, 0
                break
        elif dif[i][0] == '-':
            neg += 1
        elif dif[i][0] == '+':
            pos += 1
        

templates = pickle.load(open('autTemplatesCheckedwithSourcesNoPOS.p', 'rb'))
print(len(templates))
with open('wilkinsWithDialTurn.csv', 'r') as wilkins:
    wilkins = csv.reader(wilkins, delimiter='\t')
    for line in wilkins:
        wilks.append((line[0].split(), line[1].split(), line[3], line[2], line[2], line[4], line[5], line[6], line[-1]))

for (sent1, sent2, source1, source2) in templates:
    for (wilk0, wilk1, label, response, cs_guess, cs_correct, c_code_1, c_code_2, dialTurn) in wilks:
        w, x, y, z = doDiff(label, sent2, sent1, wilk0, source1, source2, response, cs_guess, cs_correct, c_code_1, c_code_2, dialTurn), \
                     doDiff(label, sent2, sent1, wilk1, source1, source2, response, cs_guess, cs_correct, c_code_1, c_code_2, dialTurn), \
                     doDiff(label, sent1, sent2, wilk0, source1, source2, response, cs_guess, cs_correct, c_code_1, c_code_2, dialTurn), \
                     doDiff(label, sent1, sent2, wilk1, source1, source2, response, cs_guess, cs_correct, c_code_1, c_code_2, dialTurn)
        if w is not None:
            matches.append(w)
        if x is not None:
            matches.append(x)
        if y is not None:
            matches.append(y)
        if z is not None:
            matches.append(z)
print(len(matches))
pickle.dump(matches, open('matchesNoPOS.p', 'wb'))

##test = [('lab','trans', ['$0', 'you', '$1'],['are', 'you', 'healthy']),
##        ('lab','trans',['$0', 'you', '$1'], ['what', 'do', 'you', 'do', 'for', 'a', 'living']),
##        ('lab','trans', ['$0', 'you', '$1'], ['does', 'that', 'worry', 'you']),
##        ('lab','trans', ['$0', 'you', '$1'], ['you', 'have', 'pain', 'when', 'urinating']),('lab','trans',['how', 'is', '$0', '$1'],['how', 'is', 'your', 'health']),
##        ('lab','trans',['how', 'is', '$0', '$1'],['okay', 'and', 'how', 'is', 'your', 'mother']),
##        ('lab','trans',['how', 'is', '$0', '$1'],['how', 'is', 'work']),
##        ('lab','trans',['how', 'is', '$0', '$1'],['well', 'how', 'is', 'life'])]
##for (a,b,c,d) in test:
##    doDiff(a,b,c,d)
