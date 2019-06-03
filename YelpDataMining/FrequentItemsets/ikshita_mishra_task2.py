
import time
import sys

from pyspark import SparkContext
from itertools import combinations

sc = SparkContext()


start = time.time()

datacsv = sc.textFile(sys.argv[3]).map(lambda x : x.split(","))

header  = datacsv.first()

datardd = datacsv.filter(lambda x: x != header)

kThres = int(sys.argv[1])


listOfEachBasket = datardd.map(lambda x : (str(x[0]),str(x[1]))).groupByKey().mapValues(set)
listOfEachBasket = listOfEachBasket.map(lambda x: x[1]).filter(lambda x: len(x) > kThres)

basklen = listOfEachBasket.count()
print(basklen)
'''
def f(iterator): yield list(iterator)


print(listOfEachBasket.collect())
print(partitions)
print(listOfEachBasket.mapPartitions(f).collect())
'''

support = int(sys.argv[2])

def getFrequentItems(freq, k,bask,thresSupport):
    #Check k
    #Initialize dictionary for counts
    size2 = {}
    candi = list()
    if k == 2 :
        #get combinations for k=2 only
        for v in combinations(freq, k):
            candi.append(list(v))

    elif k>=3:

        freq = list(freq)
        #create combinations using freq from k-1 pass
        for m in range(len(freq) - 1):

                for n in range(m + 1, len(freq)):

                    #give a slicing
                    endLimit = (k - 2)
                    a = freq[m][0:endLimit]
                    b = freq[n][0:endLimit]
                    #Check intersection then take Union of two sets
                    if (a == b):
                        candi.append(list(set(freq[m]).union(set(freq[n]))))
                    else:
                        break
    #Create frequent items
    res =[]
    for cd in candi:
        #Create set to check subset
        cdset = set(cd)

        for basket in bask:
            if cdset.issubset(basket):
                key = tuple(sorted(cd))
                if key in size2:
                    #Check for count if greater or equal to supp append to final list
                    if size2[key] < thresSupport:
                        size2[key]  += 1
                        if size2[key] >= thresSupport:
                            res.append(key)
                    else:
                        break

                else:
                    size2[key] = 1
                    # Check for count if greater or equal to supp append to final list
                    if size2[key] >= thresSupport:
                        res.append(key)

    return sorted(res)

def secondPhase(candidates,numBaskets):

    baskets = list(numBaskets)

    icount = {}
    for candidate in candidates:
        item = candidate

        if type(candidate) is tuple:
            cand = set([a for a in candidate])
        else:
            cand = {candidate}
            item = (candidate,)

        for basket in baskets:
            if cand.issubset(basket):
                if item in icount:
                    icount[item] += 1
                else:
                    icount[item] = 1
    return icount.items()


def AprioriAlgorithm(basklen,numBaskets, support):

    bask = list(numBaskets)
    #print(bask)
    #To have accurate partition based on ratios
    thresSupport =  (float(len(bask)) / float(basklen)) * support
    #final freq list
    resultSON = list()

    freqsize1 =list()

    size1 ={}
    for basket in bask:
        for item in basket:
            if item in size1:
                if size1[item] < thresSupport:
                    size1[item] += 1
                    if size1[item] >= thresSupport:
                        freqsize1.append(item)


            else:
                size1[item] = 1
                if size1[item] >= thresSupport:
                    freqsize1.append(item)

    freqsize1 = sorted(freqsize1)

    resultSON.extend(freqsize1)
    k = 2 #initialise
    freq = set(freqsize1)


    while len(freq) != 0:

        newFrequentItems = getFrequentItems(freq, k,bask,thresSupport)
        resultSON.extend(newFrequentItems)
        freq = list(set(newFrequentItems))
        freq.sort()


        k += 1

    return resultSON
# check =[]
# for item in listOfEachBasket.collect():
#     check.append(item)



# allItems = collect.values()
# print(allItems)

allItems = listOfEachBasket
#.values()

#Phase1 - Find Frequent Itemsets

m1 = allItems.mapPartitions(lambda numBaskets : AprioriAlgorithm(basklen,numBaskets, support)).map(lambda x : (x,1))

r1 = m1.reduceByKey(lambda a,b:a).keys().collect()


m2 = allItems.mapPartitions(lambda numBaskets : secondPhase(r1,numBaskets))

r2 = m2.reduceByKey(lambda x,y:x+y).filter(lambda x: x[1]>=support).keys().collect()


freItems = sorted(r2)



for m in range(0,len(r1)):
    if type(r1[m]) is not tuple:
        r1[m] = (r1[m],)

candItems = sorted(r1)




canLen = list(set(len(x) for x in candItems))

freLen = list(set(len(x) for x in freItems))


f = open(sys.argv[4], "w")

f.write("Candidates:")
f.write("\n")
for i in canLen:
    can = []
    for j in candItems:
        if len(j) == i:
            can.append(j)
    if i == 1:
        f.write(', '.join(map(str, ["('%s')" % x for x in can])))
    else:
        f.write(', '.join(map(str, can)))
    f.write("\n")
    f.write("\n")


f.write("Frequent Items:")
f.write("\n")
for i in freLen:
    freq = []
    for j in freItems:
        if len(j) == i:
            freq.append(j)
    if i == 1:
        f.write(', '.join(map(str, ["('%s')" % x for x in freq])))
    else:
        f.write(', '.join(map(str, freq)))
    f.write("\n")
    f.write("\n")


end = time.time()

print("Duration :" , end-start)