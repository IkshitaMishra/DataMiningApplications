import sys
from datetime import datetime, timezone
from pyspark import SparkContext
import json
import binascii
from pyspark.streaming import StreamingContext

sc = SparkContext()
sc.setLogLevel(logLevel="OFF")
hostname = "localhost"
portNum = int(sys.argv[1])
ssc = StreamingContext(sc, 10)
lines = ssc.socketTextStream(hostname, portNum)
words = lines.map(lambda x : (json.loads(x)['city']))

bitArray = [0] * 200
instream = set()
falsePos = 0
trueNeg = 0

with open(sys.argv[2], 'w') as f:
    f.write("Time,FPR")
    f.write("\n")


def txt(x):
    with open(sys.argv[2], 'a') as f:
        f.write(str(x[0]) + "," + str(x[1]))
        f.write("\n")



def bloom(lis):
    global bitArray
    global instream
    global falsePos
    global  trueNeg
    lis = lis.collect()
    sToInt = []
    for s in lis:
        sToInt.append(int(binascii.hexlify(s.encode('utf8')), 16))

    for i in sToInt:
        # f(x) = ((ax + b) % p) % m
        h1 = abs((((87 * i) + 91) % 671) % len(bitArray))
        h2 = abs((((93 * i) + 32) % 671) % len(bitArray))
        h3 = abs((((85 * i) + 95) % 671) % len(bitArray))

        a1 = bitArray[h1]
        a2 = bitArray[h2]
        a3 = bitArray[h3]

        if (a1== 0) or (a2== 0) or (a3== 0):
            if i not in instream:
                trueNeg = trueNeg + 1
            bitArray[h1] = 1
            bitArray[h2] = 1
            bitArray[h3] = 1
            instream.add(i)
        elif (a1== 1) and (a2== 1) and (a3== 1):
            if i not in instream:
                falsePos = falsePos + 1
            instream.add(i)
    if falsePos != 0 or trueNeg != 0:
        fpr = falsePos / (falsePos + trueNeg)
    elif falsePos == 0 and trueNeg == 0:
        fpr = 0.0

    calTime = datetime.utcnow().replace(tzinfo=timezone.utc).astimezone(tz=None).strftime('%Y-%m-%d %H:%M:%S')
    txt((calTime,fpr))

words.foreachRDD(lambda rdd: bloom(rdd))



ssc.start()
ssc.awaitTermination()
ssc.stop()

