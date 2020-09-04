import pickle,nltk

bestmatches = []
matches = pickle.load(open('matches.p', 'rb'))
for (label, wilkins, matched, trans) in matches:
    for i in range(1, len(matched)+1):
        if not i==len(matched):
            if not '$' in matched[i] and not '$' in matched[i-1]:
                ind = wilkins.index(matched[i-1])
                if not wilkins[ind:].index(matched[i]) == 1:
                    break
        else:
            bestmatches.append((label, wilkins, matched, trans))

print(len(bestmatches))
##
##for (label, wilkins, matched, trans) in bestmatches:
##    wilkstag = nltk.pos_tag(wilkins)
##    for i in range(0, len(matched)):
##        if '$' in matched[i] and '/' in matched[i]:
##            tag = matched[i].split('/')[1]
##            if i==0:
##                if not wilkstag[0][1] == tag:
##                    break
##            elif '$' not in matched[i-1]:
##                ind = wilkins.index(matched[i-1])+1
##                if not wilkstag[ind][1] == tag:
##                    break
##            elif '$' not in matched[i+1]:
##                ind = wilkins.index(matched[i+1])-1
##                if not wilkstag[ind][1] == tag:
##                    break
##            else:
##                j,k = i,i
##                low,high = 0,len(matched)-1
##                while j>0:
##                    if not '$' in matched[j] or ('$' in matched[j] and '/' in matched[j]):
##                        low = j
##                        break
##                    else: j-=1
##                while k<len(matched):
##                    if not '$' in matched[k] or ('$' in matched[k] and '/' in matched[k]):
##                        high = k
##                        break
##                    else: k+=1
##                
