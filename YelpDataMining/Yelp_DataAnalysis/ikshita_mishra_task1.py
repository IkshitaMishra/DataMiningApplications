from pyspark import SparkContext
import json
import sys
import time

sc = SparkContext()


start = time.time()

#Reading and Parsing JSON File

jsonrdd = sc.textFile(sys.argv[1]).\
    map(lambda x : (json.loads(x)))

jsonrdd.cache()

# Total Number of Reviews
n_reviews = jsonrdd.count()


# Total Number of Reviews in 2018
n_reviews_2018 = jsonrdd.\
    filter(lambda x: x['date'].startswith('2018')).\
    count()


# User_id : top 10 users who wrote the largest numbers of reviews
top10_users = jsonrdd.\
    map(lambda x : x['user_id']).\
    countByValue()
user = sorted(top10_users.items(), key=lambda x: (-x[1], x[0]))
# Distinct Users
n_user = len(user)


#Business_id : The top 10 businesses that had the largest numbers of reviews

top10_businesses = jsonrdd.\
    map(lambda x : x['business_id']).\
    countByValue()
bus = sorted(top10_businesses.items(), key=lambda x: (-x[1], x[0]))

#Distinct Business
n_bus = len(bus)

#Writing output to json file

data = {
    "n_review" : n_reviews,
    "n_review_2018" : n_reviews_2018,
    "n_user" : n_user,
    "top10_user" : user[:10],
    "n_business" : n_bus,
    "top10_business" : bus[:10]
}

with open(sys.argv[2], 'w') as outfile:
    json.dump(data,outfile)



end = time.time()
print(end-start)