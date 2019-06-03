from pyspark import SparkContext
import json
import time
import sys


sc = SparkContext()
start = time.time()
# City and business (review)
bussrdd = sc.textFile(sys.argv[2]).\
    map(lambda x : (json.loads(x))).\
    map(lambda x : ( x['business_id'],x['city']))
bussrdd.cache()

# Stars and business (review)
revrdd = sc.textFile(sys.argv[1]).\
    map(lambda x : (json.loads(x))).\
    map(lambda x : ( x['business_id'],x['stars']))
revrdd.cache()

# Join Business and Review File on Business ID
joinedrdd = bussrdd.join(revrdd).\
    map(lambda x:x[1]).\
    combineByKey(lambda b: (b, 1),lambda a, b: (a[0] + b, a[1] + 1),lambda a, b: (a[0] + b[0], a[1] + b[1])).\
    map(lambda k: (k[0], k[1][0] / k[1][1]))

sortedjoinedrdd = joinedrdd.\
    sortBy(lambda x : (-x[1],x[0]))

#Method 1
start1 = time.time()
print("Method1")
for line in sortedjoinedrdd.collect()[:10]:
    print(line[0])
end1 = time.time()





#Method 2
start2 = time.time()
print("Method2")
for line in sortedjoinedrdd.take(10):
    print(line[0])
end2 = time.time()



with open(sys.argv[3], 'w') as output:
    output.write('city,stars')
    for line in sortedjoinedrdd.collect():
        output.write("\n")
        output.write(line[0] +"," + str(line[1]))


data = {
    "m1": end1 - start1,
    "m2" : end2 - start2,
    "explanation" : "collect() takes much more execution time than take(), since collect() makes list of all the key-value pairs and \n take() select top n and makes list of only those n key-values. \Thus execution time of collect() is greater than take()",
}
with open(sys.argv[4], 'w') as outfile:
    json.dump(data, outfile)


end= time.time()

print(end-start)

