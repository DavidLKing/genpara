# refineMatches.py by Sarah Ewing last updated 2/22/19
# this file eliminates matches from makeMatches4.py which actually
# should not have been matches
# after this file use makePar4.py

# optionally comment out section 5 below and dump to best matches 5 for no
# POS restrictions on multiword variables

import pickle
import nltk

# this first part removes matches where two words are adjacent in the template
# but not in the Wilkins sentence
# label is wilkins.csv line[3] "gold label"
# wilkins is wilkins.csv line[0] or line[1] "observed sentence in dialogue"
# matched is the left side of the template which matched on the Wilkins sentence
# trans is the right half of that template
# the rest are just being pulled through

bettermatches = []
matches = pickle.load(open('matchesWithSources4.p', 'rb'))
for (label, wilkins, matched, trans, source1, source2, dialTurn) in matches:
    for i in range(1, len(matched)+1):
        if not i == len(matched):
            if '$' not in matched[i] and '$' not in matched[i-1]:
                ind = wilkins.index(matched[i-1])
                if not wilkins[ind:].index(matched[i]) == 1:
                    break
        else:
            bettermatches.append((label, wilkins, matched, trans, source1, source2, dialTurn))

print(len(bettermatches))
pickle.dump(bettermatches, open('betterMatchesWithSources4.p', 'wb'))

# variables are the same as above
# this section makes sure each variable is filled with the correct number of words
# with the correct POS tags

bestmatches = []

fail1 = 0
fail2 = 0
fail3 = 0
def check_match(wilkins, matched, fail1, fail2, fail3):
    wilkstag = nltk.pos_tag(wilkins)
    ind = 0
    for i in range(0, len(matched)+1):
        if i != len(matched):
            if '$' in matched[i] and '//' in matched[i]:
                ind += len(list(matched[i].split('//')[1]))
                pass
                # pos_list = list(matched[i].split('//')[1])      # SECTION 5
                # for l in range(0, len(pos_list)):
                #     if not pos_list[l] == wilkstag[ind][1]:
                #         #fail1 += 1
                #         return False
                    # else: fail1 += 1
                #     ind += 1
            elif '$' in matched[i] and '/' in matched[i]:
                tag = matched[i].split('/')[1]
                try:
                    ttag = wilkstag[ind][1]
                except IndexError:
                    return False
                if not tag == ttag:
                    #fail2 += 1
                    return False
                # else: fail2 += 1
                ind += 1
            else:
                try:
                    word = wilkstag[ind][0]
                except IndexError:
                    return False
                if not matched[i] == word:
                    #fail3 += 1
                    return False
                # else: fail3 += 1
                ind += 1
        else:
            return True


for (label, wilkins, matched, trans, source1, source2, dialTurn) in bettermatches:
            if check_match(wilkins, matched, fail1, fail2, fail3) is True:
                bestmatches.append((label, wilkins, matched, trans, source1, source2, dialTurn))

print(len(bestmatches))

for (label, wilkins, matched, trans, source1, source2, dialTurn) in bettermatches:
    if not (label, wilkins, matched, trans, source1, source2, dialTurn) in bestmatches:
        wilkstag = nltk.pos_tag(wilkins)
        match = 0
        for item in matched:
            if len(item.split('//')) > 1:
                match += len(list(item.split('//')[1]))
            else:
                match += 1
        if len(matched[0].split('//')) > 1:
            tag = list(matched[0].split('//')[1])[0]
            for l in range(0, len(wilkstag)):
                if wilkstag[l][1] == tag:
                    ind = l
                    break
            if not ind + match > len(wilkstag):
                if check_match(wilkins, matched, fail1, fail2, fail3) is True:
                    bestmatches.append((label, wilkins, matched, trans, source1, source2, dialTurn))
        elif len(matched[0].split('/')) > 1:
            tag = matched[0].split('/')[1]
            for l in range(0, len(wilkstag)):
                if wilkstag[l][1] == tag:
                    ind = l
                    break
            if not ind + match > len(wilkstag):
                if check_match(wilkins, matched, fail1, fail2, fail3) is True:
                    bestmatches.append((label, wilkins, matched, trans, source1, source2, dialTurn))
        else:
            for l in range(0, len(wilkstag)):
                if wilkstag[l][0] == matched[0]:
                    ind == l
                    break
            if not ind + match > len(wilkstag):
                if check_match(wilkins, matched, fail1, fail2, fail3) is True:
                    bestmatches.append((label, wilkins, matched, trans, source1, source2, dialTurn))

print(len(bestmatches))
print('fail1', fail1)
print('fail2', fail2)
print('fail3', fail3)
pickle.dump(bestmatches, open('bestMatchesWithSources5.p', 'wb'))
