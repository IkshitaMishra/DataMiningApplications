import random
import sys
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans
import numpy as np
import time
start = time.time()


# ====================== Reading Textfile and Shuffle ===============

txtdata = open(sys.argv[1], 'r')
data = [line.rstrip().split(',') for line in txtdata.readlines()]
data = [[float(j) for j in i] for i in data]
dimData = [[r for col, r in enumerate(rows) if (col!= 1)] for rows in data]
random.shuffle(dimData)

indexes = [row[0] for row in data]
groundTruthClusterPoints = [row[1] for row in data]

point_dim = {}
for i in dimData:
    point_dim[tuple(i[1:])] = i[0]


# ============================ Step 1  : Divide into chunks of 20% of data =========

end = int(0.2 * len(dimData))
def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

chunk_data = list(chunks(dimData, end))



#============================== Step 2 : (Run K-means on large K) =============================

dt = chunk_data[0]

firstPass =[]
for i in dt:
    firstPass.append(i[1:])

X = np.array(firstPass)
clusNum = int(10 * int(sys.argv[2]))
kmeans = KMeans(n_clusters=clusNum, random_state=0).fit(X)
res = zip(kmeans.labels_,X.tolist())


clusterData = {}
for key, value in sorted(res):
    if key not in clusterData:
        clusterData[key] = []
    clusterData[key].append(value)

#============================== Step 3 : Create RS with one data point as cluster =================

RS_li = []
remaining_data = []
for i in clusterData:
    if len(clusterData[i]) == 1:
        RS_li.append(clusterData[i][0])
    elif len(clusterData[i]) > 1:
        remaining_data.extend(clusterData[i])

# =============================== Step 4 : Run K-mean on rest of data points for DS Cluster====================

X_2 = np.array(remaining_data)
cl =int(sys.argv[2])
kmeans_2 = KMeans(n_clusters=cl, random_state=0).fit(X_2)
res_ds = zip(kmeans_2.labels_, X_2.tolist())

dsp = {}
for key, value in sorted(res_ds):
    if key not in dsp:
        dsp[key] = []
    dsp[key].append(value)



# =============================== Step 5 : DS Summarization ====================
#[POINTS,SUM/N,SUMSQ/N,STD]
summDS = {}
for key,value in dsp.items():
    if key not in summDS:
        summDS[key] = [[],[],[],[]]
    summDS[key][0] = [point_dim[tuple(x)] for x in value]
    N = len(value)
    summDS[key][1] = (np.sum(np.array(value), axis=0)).tolist()
    summDS[key][2] = (np.sum(np.square(np.array(value)), axis=0)).tolist()
    summDS[key][3] = (np.sqrt(np.subtract(np.divide(summDS[key][2],N),np.square(np.divide(summDS[key][1],N))))).tolist()


#=============================== Step 6 : Run K-means on RS to form RS and CS ====================

if len(RS_li) > 2:
    retCluster = (int(len(RS_li) / 2)) + 1
elif len(RS_li) <= 2:
    retCluster = 1
X_3 = np.array(RS_li)
kmeans_3 = KMeans(n_clusters=retCluster, random_state=0).fit(X_3)
res_3 = zip(kmeans_3.labels_,X_3.tolist())

clusterData2 = {}
for key, value in sorted(res_3):
    if key not in clusterData2:
        clusterData2[key] = []
    clusterData2[key].append(value)


#============================== Step 3 : Create RS with one data point as cluster =================

new_retained = []
summCS = {}
for key,value in clusterData2.items():
    if len(value) == 1:
        new_retained.append(value[0])
    elif len(value) > 1:
        if key not in summCS:
            summCS[key] = [[], [], [], []]
        summCS[key][0] = [point_dim[tuple(x)] for x in value]
        N = len(value)
        summCS[key][1] = (np.sum(np.array(value), axis=0)).tolist()
        summCS[key][2] = (np.sum(np.square(np.array(value)), axis=0)).tolist()
        summCS[key][3] = (np.sqrt(np.subtract(np.divide(summCS[key][2], N), np.square(np.divide(summCS[key][1], N))))).tolist()


dsLen = 0
for i in summDS:
    dsLen = dsLen + len(summDS[i][0])
csLen = 0
for i in summCS:
    csLen = csLen + len(summCS[i][0])

f = open(sys.argv[3], "w")
f.write("The intermediate results:")
f.write("\n")
f.write("Round 1: " + str(dsLen) + "," + str(len(summCS))+ "," + str(csLen) + "," +str(len(new_retained)))
f.write("\n")


dim = len(firstPass[0])
threshold = 2 * np.sqrt(dim)
count = 1


lastOne = len(chunk_data) - 1

csKeys = sorted(summCS.keys())
for i in chunk_data[1:]:
    count += 1
    numlen = len(i)
    #print("Round :", count)
    for dpt in i:
        pointNo = dpt[0]
        datapoint = dpt[1:]
        flag= -1
        if flag == -1:
            li = []
            for key,value in summDS.items():
                tot = len(value[0])
                diff = np.subtract(datapoint,np.divide(value[1],tot))
                ans = np.divide(diff, value[3])
                fin = np.sqrt(np.sum(np.square(ans)))
                li.append((key, fin))

            li = sorted(li, key=lambda x: x[1])

            for m in li:
                if m[1] < threshold:
                    ind = m[0]
                    flag = 0
                    summDS[ind][0].append(pointNo)
                    N = len(summDS[ind][0])
                    sumarr = [summDS[ind][1]] + [datapoint]
                    summDS[ind][1] = (np.sum(np.array(sumarr), axis=0)).tolist()
                    summDS[ind][2] = (np.sum([summDS[ind][2],np.square(np.array(datapoint))], axis=0)).tolist()
                    summDS[ind][3] = (np.sqrt(np.subtract(np.divide(summDS[ind][2],N),np.square(np.divide(summDS[ind][1],N))))).tolist()
                    break

        if flag == -1:

            CSli = []

            for key,value in summCS.items():
                tot = len(value[0])
                diff = np.subtract(datapoint,np.divide(value[1],tot))
                ans = np.divide(diff, value[3])
                ans = np.sqrt(np.sum(np.square(ans)))
                CSli.append((key, ans))
            CSli = sorted(CSli, key=lambda x: x[1])
            for m in CSli:
                if m[1] < threshold:
                    ind = m[0]
                    flag = 0
                    summCS[ind][0].append(pointNo)
                    N = len(summCS[ind][0])
                    sumarr = [summCS[ind][1]] + [datapoint]
                    summCS[ind][1] = (np.sum(np.array(sumarr), axis=0)).tolist()
                    summCS[ind][2] = (np.sum([summCS[ind][2],np.square(np.array(datapoint))], axis=0)).tolist()
                    summCS[ind][3] = (np.sqrt(np.subtract(np.divide(summCS[ind][2],N),np.square(np.divide(summCS[ind][1],N))))).tolist()
                    break

        if flag == -1:
            flag = 0

            new_retained.append(datapoint)


    #print("Point Added")
    #print(len(new_retained))
    ##=============================== Step 11 : Run K-means on RS to form RS and CS ====================
    if len(new_retained) > 0:
        if len(new_retained) > 2:
            retCluster = (int(len(new_retained) /2 )) + 1
        elif len(new_retained) <= 2:
            retCluster = 1
        X_4 = np.array(new_retained)
        kmeans_4 = KMeans(n_clusters=retCluster, random_state=0).fit(X_4)
        res_4 = zip(kmeans_4.labels_, X_4.tolist())

        clusterData3 = {}
        for key, value in sorted(res_4):
            if key not in clusterData3:
                clusterData3[key] = []
            clusterData3[key].append(value)

        new_retained = []
        for i in clusterData3:
            if len(clusterData3[i]) == 1:
                new_retained.append(clusterData3[i][0])
            elif len(clusterData3[i]) > 1:
                newKey = csKeys[-1] + 1
                value = clusterData3[i]
                csKeys.append(newKey)
                if newKey not in summCS:
                    summCS[newKey] = [[], [], [], []]
                summCS[newKey][0] = [point_dim[tuple(x)] for x in value]
                N = len(value)
                summCS[newKey][1] = (np.sum(np.array(value), axis=0)).tolist()
                summCS[newKey][2] = (np.sum(np.square(np.array(value)), axis=0)).tolist()
                summCS[newKey][3] = (np.sqrt(np.subtract(np.divide(summCS[newKey][2], N), np.square(np.divide(summCS[newKey][1], N))))).tolist()

    iterList = list(summCS.keys())

    for i in range(0, len(iterList)):
        liAns = []
        for j in range(i + 1, len(iterList)):
            old = iterList[i]
            new = iterList[j]
            tot_old = len(summCS[old][0])
            tot_new = len(summCS[new][0])
            diff = np.subtract(np.divide(summCS[old][1],tot_old),np.divide(summCS[new][1],tot_new))
            ans = np.divide(diff, summCS[new][3])
            ans = np.sqrt(np.sum(np.square(ans)))
            liAns.append((new, ans))
        liAns = sorted(liAns, key=lambda x: x[1])
        for m in liAns:
            if m[1] < threshold:
                ind = m[0]
                old = iterList[i]

                summCS[ind][0].extend(summCS[old][0])
                N = len(summCS[ind][0])
                sumarr = [summCS[ind][1]] + [summCS[old][1]]
                summCS[ind][1] = (np.sum(np.array(sumarr), axis=0)).tolist()
                summCS[ind][2] = (np.sum([summCS[ind][2], np.square(np.array(summCS[old][2]))], axis=0)).tolist()
                summCS[ind][3] = (np.sqrt(np.subtract(np.divide(summCS[ind][2], N), np.square(np.divide(summCS[ind][1], N))))).tolist()
                del summCS[old]

                break

    if count == lastOne:
        iterListCS = list(summCS.keys())
        iterListDS = list(summDS.keys())

        for i in iterListCS:

            liAns = []
            for j in iterListDS:
                old = i
                new = j
                tot_old = len(summCS[old][0])
                tot_new = len(summDS[new][0])
                diff = np.subtract(np.divide(summCS[old][1], tot_old), np.divide(summDS[new][1], tot_new))
                ans = np.divide(diff, summDS[new][3])
                ans = np.sqrt(np.sum(np.square(ans)))
                liAns.append((new, ans))
            liAns = sorted(liAns, key=lambda x: x[1])
            flag = -1
            if flag == -1:
                for m in liAns:
                    if m[1] < threshold:
                        flag = 0
                        ind = m[0]
                        old = i
                        summDS[ind][0].extend(summCS[old][0])
                        N = len(summDS[ind][0])
                        sumarr = [summDS[ind][1]] + [summCS[old][1]]
                        summDS[ind][1] = (np.sum(np.array(sumarr), axis=0)).tolist()
                        summDS[ind][2] = (
                            np.sum([summDS[ind][2], np.square(np.array(summCS[old][2]))], axis=0)).tolist()
                        summDS[ind][3] = (np.sqrt(np.subtract(np.divide(summDS[ind][2], N),
                                                              np.square(np.divide(summDS[ind][1], N))))).tolist()
                        del summCS[old]
                        break

    dsLenA = 0
    for i in summDS:
        dsLenA = dsLenA + len(summDS[i][0])

    csLenA = 0
    for i in summCS:
        csLenA = csLenA + len(summCS[i][0])

    f.write("Round " + str(count) + ": " + str(dsLenA) + "," + str(len(summCS)) + "," + str(csLenA) + "," + str(
        len(new_retained)))
    f.write("\n")






f.write("\n")
f.write("The clustering results:")
f.write("\n")

createDs={}
for i in summDS:
    m = summDS[i][0]
    for j in m:
        createDs[j] = i

chk=[]
for i in indexes:
    if i in createDs:
        f.write(str(int(i))+","+str(createDs[i]))
        chk.append(createDs[i])
        f.write("\n")
    if i not in createDs:
        f.write(str(int(i))+","+str(-1))
        f.write("\n")
        chk.append(-1)





s =normalized_mutual_info_score(chk, groundTruthClusterPoints)
print("Accuracy :",s)
#print("Percentage of discard points 0.95* Data:",dsLenA,(dsLenA > (0.95*len(dimData))))
end =time.time()
print("Duration:",end-start)


"""


s =normalized_mutual_info_score(chk, groundTruthClusterPoints)
print("Accuracy :",s, s*100)
print("Percentage of discard points :",dsLenA,len(dimData),(dsLenA > (0.95*len(dimData))))
=========================== Testing ========================

1) DISCARD POINTS IN LAST ROUND > .95 * entire data
2) ACCURACY

"""