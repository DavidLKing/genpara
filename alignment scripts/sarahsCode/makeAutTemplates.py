import pickle
import csv

nums = []
for i in range(0, 191):
    num = ''
    for j in range(0, 4-len(list(str(i)))):
        num += '0'
    num += str(i)
    nums.append(num)
    
stopwords = {'is', 'are', 'what', 'who', 'where', 'when', 'how', 'why', 'you', 'your', 'have', 'has', 'be', 'to', 'them'}

templates = []
for item in nums:
    filename = 'goldalign-repo-master/data/users/arbit/demo-user-1_demo-user-2/complete/vpd-corpus/batch_'+item+'.tsv'
    try:
        with open(filename, 'r') as file:
            read = csv.reader(file, delimiter='\t')
            for i, line in enumerate(read):
                if line[6] == '1':
                    aligns, aligns2, finalaligns = [],[],[]
                    sent1 = line[1].split()
                    sent2 = line[3].split()
                    sure = line[7].split()
                    poss = line[8].split()
                    align = sure+poss
                    for item in align:
                        x,y = item.split('-')
                        aligns.append(([int(x)],[int(y)]))
                    for pair in align:
                        x,y = pair.split('-')
                        for item in aligns:
                            if int(x) in item[0]:
                                if not int(y) in item[1]:
                                    item[1].append(int(y))
                            elif int(y) in item[1]:
                                item[0].append(int(x))
                    for item in aligns:
                        left = sorted(item[0])
                        right = sorted(item[1])
                        if not (left, right) in aligns2:
                            aligns2.append((left, right))
                    for (left,right) in aligns2:
                        l,r = 0,0
                        for i in left:
                            if not sent1[i] in stopwords: l+=1
                        for i in right:
                            if not sent2[i] in stopwords: r+=1
                        if l > 0 or r > 0:
                            finalaligns.append((left,right))
                    for i in range(0, len(finalaligns)):
                        tpl = finalaligns[i]
                        left,right = tpl[0],tpl[1]
                        j,k = 0,0
                        while j < len(left):
                            if j==0:
                                sent1[left[j]] = "$"+str(i)
                                j+=1
                            else:
                                sent1[left[j]] = "REPLACE"
                                j+=1
                        while k < len(right):
                            if k==0:
                                sent2[right[k]] = "$"+str(i)
                                k+=1
                            else:
                                sent2[right[k]] = "REPLACE"
                                k+=1
                    while "REPLACE" in sent1: sent1.remove("REPLACE")
                    while "REPLACE" in sent2: sent2.remove("REPLACE")
                    i,j = 0,0
                    for word in sent1:
                        if not "$" in word:
                            i+=1
                    if i!= 0 and i!= len(sent1):
                        for word in sent2:
                            if not "$" in word:
                                j+=1
                        if j!=0 and j!= len(sent2):
                            if not sent1==sent2:
                                templates.append((sent1, sent2, line[1], line[3]))
##                    print('Align', align)
##                    print('Aligns2', aligns2)
##                    print('Finalaligns', finalaligns)
##                    print("Sent1", sent1)
##                    print('Sent2', sent2)
##                    print('Templates', templates)
##                    print()
    except FileNotFoundError:
        pass

##for item in templates:
##    print(item)
##print(len(templates))
pickle.dump(templates, open('autTemplateswithSourcesNoPOS.p', 'wb'))
