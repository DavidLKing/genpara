import pickle, nltk

bettermatches = []
matches = pickle.load(open('matchesWithFullSources.p', 'rb'))
for (label, wilkins, matched, trans, source1, source2, response, cs_guess, cs_corr, cc1, cc2, dialTurn) in matches:
    for i in range(1, len(matched)+1):
        if not i == len(matched):
            if '$' not in matched[i] and '$' not in matched[i-1]:
                ind = wilkins.index(matched[i-1])
                if not wilkins[ind:].index(matched[i]) == 1:
                    break
        else:
            bettermatches.append((label, wilkins, matched, trans, source1, source2, response, cs_guess, cs_corr, cc1, cc2, dialTurn))

print(len(bettermatches))
pickle.dump(bettermatches, open('betterMatchesWithFullSources.p', 'wb'))

bestmatches = []
for (label, wilkins, matched, trans, source1, source2, response, cs_guess, cs_corr, cc1, cc2, dialTurn) in bettermatches:
    wilkstag = nltk.pos_tag(wilkins)
    for i in range(0, len(matched)+1):
        if not i==len(matched):
            if '$' in matched[i] and '/' in matched[i]:
                tag = matched[i].split('/')[1]
                if i==0:                                    ##case1
                    if not wilkstag[0][1] == tag:
                        break
                    else:
                        last = 0
                elif '$' not in matched[i-1]:               ##case2
                    ind = wilkins.index(matched[i-1])+1
                    if not wilkstag[ind][1] == tag:
                        break
                    else:
                        last = ind
                elif i<len(matched)-1 and'$' not in matched[i+1]:               ##case3
                    ind = wilkins.index(matched[i+1])-1
                    if not wilkstag[ind][1] == tag:
                        break
                    else:
                        last = ind
                elif ('$' in matched[i-1] and '/' in matched[i-1]):     ##case4
                    if not wilkstag[last+1][1] == tag:
                        break
                    else:
                        last = ind
                else:
                    b,f = i-1,i+1
                    single = 0
                    while b>-1:
                        if not '$' in matched[b] or ('$' in matched[b] and '/' in matched[b]):
                            break
                        else: b-=1
                    while f<len(matched):
                        if not '$' in matched[f]:
                            break
                        elif '/' in matched[f]:
                            single+=1
                            f+=1
                        else:
                            f+=1
                    if b==-1 and f==len(matched):
                        print('wilks', wilkins)
                        print('matched', matched)
                        print()
                    if f-single == i+1:
                        if f == len(matched):
                            ind = -(single+1)
                        else:
                            ind = wilkins.index(matched[f])-(single+1)
                        if not wilkstag[ind][1] == tag:
                            break
                        else:last=ind
                    else:
                        tags = []
                        if not b == -1:
                            beg = wilkins.index(matched[b])
                        else:
                            beg = 0
                        if not f == len(matched):
                            end = wilkins.index(matched[f])
                        else:
                            end = len(wilkins)
                        for x in range(beg, end):
                            tags.append(wilkstag[x][1])
                        if not tag in tags:
                            break
                        else:
                            last = ind
        else:
            bestmatches.append((label, wilkins, matched, trans, source1, source2, response, cs_guess, cs_corr, cc1, cc2, dialTurn))

print(len(bestmatches))
pickle.dump(bestmatches, open('bestMatchesWithFullSources.p', 'wb'))
