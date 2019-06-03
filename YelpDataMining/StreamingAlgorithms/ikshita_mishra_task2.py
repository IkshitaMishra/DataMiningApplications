import math
from datetime import datetime, timezone
from pyspark import SparkContext
import json
import sys
import binascii
from pyspark.streaming import StreamingContext

sc = SparkContext()
sc.setLogLevel(logLevel="OFF")

hostname = "localhost"
portNum = int(sys.argv[1])
ssc = StreamingContext(sc, 5)
lines = ssc.socketTextStream(hostname, portNum)

words = lines.map(lambda x : (json.loads(x)['city']))


with open(sys.argv[2], 'w') as f:
    f.write("Time,Gound Truth,Estimation")
    f.write("\n")


def txt(x):
    with open(sys.argv[2], 'a') as f:
        f.write(str(x[0]) + "," + str(x[1]) + "," + str(x[2]))
        f.write("\n")


def flajolet(partition):

    lis = partition.collect()
    groundSet = set()
    sToInt = []
    for s in lis:
        sToInt.append(int(binascii.hexlify(s.encode('utf8')), 16))
        groundSet.add(s)
    hashFunc = [[87, 91, 671], [123, 192, 671], [35, 50, 671], [195, 164, 671],
                [136, 172, 671], [13, 19, 671], [93, 32, 671], [85, 95, 671], [19, 23, 671]]

    hashBit = [0] * len(hashFunc)

    m = len(sToInt)
    for num in range(0, len(hashFunc)):
        h = hashFunc[num]
        r = []
        for i in sToInt:
            hashVal = ((h[0] * i + h[1]) % h[2]) % m
            an = "{0:b}".format(hashVal)
            val = len(an) - len(an.rstrip('0'))
            if val != len(an):
                r.append(val)
            elif val == len(an):
                r.append(0)
        hashBit[num] = max(r)
    a1 = 0
    for i in range(0,3):
        a1 = a1 + 2 ** (hashBit[i])
    a1 = a1 / 3

    b1 = 0
    for j in range(3, 6):
        b1 = b1 + 2 ** (hashBit[j])
    b1 = b1 / 3

    c1 = 0
    for j in range(6, 9):
        c1 = c1 + 2 ** (hashBit[j])
    c1 = c1 / 3
    liM = sorted([a1, b1,c1])
    predictedCount = math.floor(liM[1])

    calTime = datetime.utcnow().replace(tzinfo=timezone.utc).astimezone(tz=None).strftime('%Y-%m-%d %H:%M:%S')
    a = (calTime,len(groundSet),predictedCount)
    txt(a)

words.window(30,10).foreachRDD(lambda rdd: flajolet(rdd))

ssc.start()
ssc.awaitTermination()
ssc.stop()
