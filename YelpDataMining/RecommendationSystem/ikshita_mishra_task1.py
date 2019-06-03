import sys

from pyspark import SparkConf, SparkContext
import time

#sc = SparkContext(appName='lsh', conf=SparkConf())

conf = SparkConf().setAppName("lsh")
conf = (conf
        .set('spark.executor.memory', '4G')
        .set('spark.driver.memory', '4G')
        )
sc = SparkContext(conf=conf)
sc.setLogLevel("WARN")
start = time.time()
start1 = time.time()

#Read Data, Remove Headers
csvrdd = sc.textFile(sys.argv[1])
header = csvrdd.first()
csvrdd = csvrdd.filter(lambda line:line!=header).map(lambda x: x.split(","))


"""
#Map buissness id string with number
bussdict={}
buss = csvrdd.map(lambda a: (a[1],a[0])).reduceByKey(lambda a,b:a).keys().collect()
for i,j in enumerate(sorted(buss)):
    bussdict[j] = i
"""


#Map user id string with number
userdict={}
user = csvrdd.map(lambda a: (a[0],a[1])).reduceByKey(lambda a,b:a).keys().collect()
for i,j in enumerate(sorted(user)):
    userdict[j] = i


#print(userdict)

#Groupbykey Business
busssimdata = csvrdd.map(lambda x : (x[1],userdict[x[0]])).groupByKey().mapValues(set).sortBy(lambda x:x[0])

#print(busssimdata.take(100))
bussusermappings = busssimdata.collect()



m = len(userdict)
#print(busssimdata.take(10))
#print("m", m)
bususermap ={}
for i in bussusermappings:
    bususermap[i[0]] = i[1]
"""

print(bususermap['1196'])
print(bususermap['1198'])
print((bususermap['1196']).intersection(bususermap['1198']))
print((bususermap['1196']).union(bususermap['1198']))
print(len((bususermap['1196']).intersection(bususermap['1198'])))
print(len((bususermap['1196']).union(bususermap['1198'])))
print(len((bususermap['1196']).intersection(bususermap['1198']))/len((bususermap['1196']).union(bususermap['1198'])))

"""

def minhash(userids):
    minhas = [userids[0]]
    finalhash = []
    hashfunc = \
        [[87, 91, 671], [123, 192, 671], [35, 50, 671], [195, 164, 671], [32, 37, 671], [43, 21, 671], [51, 58, 671],
         [68, 73, 671],
         [42, 78, 671], [136, 172, 671], [13, 19, 671], [93, 32, 671], [85, 95, 671], [19, 23, 671], [20, 27, 671],
         [10, 29, 671],
         [387, 552, 671], [11, 13, 671], [17, 29, 671], [53, 34, 671], [17, 35, 671], [77, 44, 671], [14, 93, 671],
         [40, 82, 671],
         [62, 87, 671], [73, 23, 671], [97, 5, 671], [17, 54, 671], [14, 83, 671], [130, 120, 671]]
    for each in hashfunc:
        selectmin = []
        for user in userids[1]:
            selectmin.append(((each[0]*user+each[1])%each[2])%m)
        finalhash.append(min(selectmin))
    minhas.append(finalhash)
    #print(minhas)
    return minhas

sigmatrix = busssimdata.map(lambda a: (minhash(a)))

#print("sig",len(sigmatrix.collect()))
#print(sigmatrix.take(10))


bands = 15

def createbands(tups):
    numhash = len(tups[1])
    rows = int(numhash / bands)
    return [(((tups[1][i], tups[1][i + 1]),i),tups[0]) for i in range(0, bands, rows)]

bandData = sigmatrix.flatMap(lambda x: createbands(x)).groupByKey().mapValues(lambda x:sorted(list(x))).filter(lambda c:len(c[1]) >= 2)

#print(bandData.take(10))
#Creating Pairs
#print("bands created")


def createCandidates(busList):
    return {((busList[1][i], busList[1][j]),1) for i in range(len(busList[1])) for j in range(i + 1, len(busList[1]))}

candPairs = bandData.flatMap(lambda x: createCandidates(x)).reduceByKey(lambda c,d:c+d).keys()
    #.groupByKey().sortBy(lambda x:x[0])

#print("cand",candPairs.take(10))
#print("cands created")
#print(candPairs.count())


def jaccardsim(candPairs):
    intersect = (bususermap[candPairs[0]]).intersection(bususermap[candPairs[1]])
    uni = (bususermap[candPairs[0]]).union(bususermap[candPairs[1]])
    return len(intersect) / len(uni)

simbus = candPairs.map(lambda x:(x[0],x[1],jaccardsim(x))).filter(lambda a:a[2]>=0.5)

f = open(sys.argv[2], "w")
f.write("business_id_1, business_id_2, similarity")
f.write("\n")
simbuslist = sorted(simbus.collect())

for i in simbuslist:
    f.write(i[0]+","+i[1]+","+str(i[2]))
    f.write("\n")


end = time.time()
print("Duration", end-start)
"""


============= Calculation of Precision and Recall ===============

predictedres = simbus.map(lambda x: (x[0],x[1]))
actualdata = sc.textFile("pure_jaccard_similarity.csv")
header2 = actualdata.first()
actualdata = actualdata.filter(lambda line:line!=header2).map(lambda x: x.split(",")).map(lambda x:(x[0],x[1]))


falsepos = predictedres.subtract(actualdata).count()
falseneg = actualdata.subtract(predictedres).count()
truepos = predictedres.intersection(actualdata).count()

final_precision = float(truepos/(truepos+falsepos))
final_recall = float(truepos/(truepos+falseneg))

print("Precision", final_precision)


print("Recall", final_recall)

end1 = time.time()




"""

