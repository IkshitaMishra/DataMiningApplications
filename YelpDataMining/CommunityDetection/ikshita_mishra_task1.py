import os
from pyspark import SparkContext, SparkConf
from graphframes import *
from itertools import combinations, chain
from pyspark.sql.types import *
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import pyspark.sql.functions as f

sc = SparkContext()
import sys
import time
spark = SparkSession(sc)
sc.setLogLevel("WARN")
os.environ["PYSPARK_SUBMIT_ARGS"] = ("--packages graphframes:graphframes:0.6.0-spark2.3-s_2.11")

start = time.time()
csvrdd = sc.textFile(sys.argv[2])
header = csvrdd.first()
csvrdd = csvrdd.filter(lambda line:line!=header).map(lambda x: x.split(","))
rdd = csvrdd.map(lambda a:(str(a[1]),str(a[0]))).groupByKey().mapValues(set)
#print(rdd.take(10))

threshold = int(sys.argv[1])
def filFunc(c):
    c = sorted(c)
    combs = combinations(c, 2)
    res =[]
    for user_pair in combs:
         res.append((user_pair, 1))
    return res

filterPhaseA = rdd.flatMap(lambda c:filFunc(c[1])).reduceByKey(lambda a,b:a+b).filter(lambda x:x[1]>=threshold).map(lambda x:x[0])
edgedf = filterPhaseA.toDF(["src","dst"])

filterPhaseB = filterPhaseA.map(lambda x:(x[1],x[0]))

createEdges = (filterPhaseA.union(filterPhaseB)).groupByKey().mapValues(set)
nodes = createEdges.keys()
userdf = spark.createDataFrame(nodes, StringType()).selectExpr("value as id")

#print(filterPhaseA.count())
#print(nodes.count())


g = GraphFrame(userdf, edgedf)
result = g.labelPropagation(maxIter=5)
community = result.groupby("label").agg(f.collect_list("id").alias("id"))
fin = community.select("id").rdd.flatMap(lambda x: x).collect()

li =[]
for i in fin:
    li.append(sorted(i))
#print(li)
reslen = sorted(list(set(len(x) for x in li)))
f = open(sys.argv[3], "w")
for i in reslen:
    can = []
    for j in li:
        if len(j) == i:
            can.append(j)
    can = sorted(can)
    for m in can:

        f.write(str(m).replace("[", "").replace("]", ""))
        f.write("\n")



end = time.time()

print("Duration", end-start)

