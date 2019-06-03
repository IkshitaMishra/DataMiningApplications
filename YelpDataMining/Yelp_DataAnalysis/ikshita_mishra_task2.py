from pyspark import SparkContext
import json
import time
import sys

sc = SparkContext()

start = time.time()

#Read Review JSON File
revrdd = sc.textFile(sys.argv[1]).\
    map(lambda x : (json.loads(x)['business_id']))

revrdd.cache()

#Default Partition ---------------------------------------

#Method1
start1 = time.time()
bus = sorted(revrdd.countByValue().items(), key=lambda x: (-x[1], x[0]))
end1 = time.time()

#Customized Partition --------------------------------------------

revrdd2 = revrdd.repartition(int(sys.argv[3]))


#Method2
start2 = time.time()
bus1 = sorted(revrdd2.countByValue().items(), key=lambda x: (-x[1], x[0]))
end2 = time.time()

# MapPartition
def f(iterator): yield len(list(iterator))

defaultpart = revrdd.mapPartitions(f).collect()


#Write to JSON File

data = {
    "default": {
        "n_partition": len(defaultpart),
        "n_items": defaultpart ,
        "exe_time": end1 - start1
    },
    "customized":{
        "n_partition": int(sys.argv[3]),
        "n_items": revrdd2.mapPartitions(f).collect(),
        "exe_time":  end2 - start2

    },
    "explanation" : "Increasing reasonable number of partition in customized function than default partition helps in load balancing i.e. proper utilization of cores and avoids overhead, running more parallel map-reduce implementations. But too many partitions or too less do not utilize core properly and also increases execution time ( less time)"

}


with open(sys.argv[2], 'w') as outfile:
    json.dump(data, outfile)


end = time.time()
print(end-start)