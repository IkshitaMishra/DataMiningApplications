import pandas as pd
import warnings
import time
import sys

from pyspark import SparkContext

sc = SparkContext()
sc.setLogLevel("WARN")
warnings.filterwarnings('ignore')
from surprise import Reader, Dataset, SVD, BaselineOnly

start = time.time()
dataset_train = pd.read_csv(sys.argv[1]+"yelp_train.csv")
dataset_train.columns = ['user_id', 'business_id', 'stars']
dataset_test = pd.read_csv(sys.argv[2])
dataset_test.columns = ['user_id', 'business_id', 'stars']

common = dataset_train.merge(dataset_test, on=['user_id', 'business_id'])
trainset = dataset_train[
    (~dataset_train.user_id.isin(common.user_id)) & (~dataset_train.business_id.isin(common.business_id))]
reader = Reader(rating_scale=(1, 5))
trainset = Dataset.load_from_df(trainset[['user_id', 'business_id', 'stars']], reader)

dataset_test = dataset_test.apply(tuple, axis=1).tolist()
trainset = trainset.build_full_trainset()

"""
'method': 'als',
               'n_epochs': 92,
               'reg_u': 6,
               'reg_i': 4   

n_epochs=75, lr_all=0.005, reg_all=0.3 72,0.005,0.23,145 (55,0.005,0.21,105) (35,0.004,0.18,73)  
        
"""

bsl_options = {'method': 'als', 'n_epochs': 94, 'reg_u': 6, 'reg_i': 4}
algo1 = BaselineOnly(bsl_options=bsl_options)
predictions1 = algo1.fit(trainset)
predictions1 = predictions1.test(dataset_test)
predList1 = []
for i in predictions1:
    predList1.append((i[0], i[1], i[2], i[3]))
resdf1 = pd.DataFrame(list(predList1), columns=['user_id', 'business_id', 'stars', 'est_stars1'])

algo2 = SVD(n_factors=105, n_epochs=55, lr_all=0.005, reg_all=0.22)
predictions2 = algo2.fit(trainset)
end = time.time()
predictions2 = predictions2.test(dataset_test)
predList2 = []
for i in predictions2:
    predList2.append((i[0], i[1], i[2], i[3]))
resdf2 = pd.DataFrame(list(predList2), columns=['user_id', 'business_id', 'stars', 'est_stars2'])

findf = pd.merge(resdf1, resdf2, how='left', on=['user_id', 'business_id', 'stars'])
findf['est_ratings'] = (findf.est_stars1 * 0.9) + (findf.est_stars2 * 0.1)
findf['absdiff'] = abs(findf.stars - findf.est_ratings)

li = []
for i, r in findf.iterrows():
    tup = (r.user_id, r.business_id, r.est_ratings, r.absdiff)
    li.append(tup)
resA = sc.parallelize(li)
absdiff = resA.map(lambda x: x[3])


def level(absdif):
    if (absdif >= 0 and absdif < 1):
        return (0, 1)
    elif (absdif >= 1 and absdif < 2):
        return (1, 1)
    elif (absdif >= 2 and absdif < 3):
        return (2, 1)
    elif (absdif >= 3 and absdif < 4):
        return (3, 1)
    elif (absdif >= 4):
        return (4, 1)


resdiff = absdiff.map(lambda x: level(x)).reduceByKey(lambda x, y: x + y).sortByKey().collect()
rootmeansqerr = pow(absdiff.map(lambda x: x * x).mean(), 0.5)



f = open(sys.argv[3], "w")
f.write("user_id, business_id, prediction")
f.write("\n")
predList = resA.collect()
for i in predList:
    f.write(str(i[0]) + "," + str(i[1]) + "," + str(i[2]))
    f.write("\n")

end = time.time()
f1 = open("ikshita_mishra_description.txt", "w")
f1.write("Method Description:")
f1.write("\n")
f1.write(
    "The objective of this project was to implement both collaborative filtering, “Alternating Least Squares (ALS) Matrix Factorizationa and Singular Value Decomposition (SVD) Matrix Factorization recommendation system. Both the algorithm are used to compute low-rank matrix factorization. SVD gives a value of zero to unknown entries. In the project, the recommender predicts the unknown business rating from a user. I calculated the weighted scores of predicted rating of intersected user-business candidates between both the recommender models. Thus, made a Hybrid Recommendation of ALS and SVD with Root Mean Square of " + str(
        rootmeansqerr) + ", which is lower than < 1.0 baseline. I used surprise python package for this implementation. The two modules used were Surprise BaselineOnly= “als” with parameters 'n_epochs': 94,'reg_u': 6,'reg_i': 4. These were given intuitively to increase the accuracy and lower the rise value. The other approach was SVD with parameters n_factors=105,n_epochs=55, lr_all=0.005, reg_all=0.21. The weighted average of the both model were In ratio is (als:svd) 0.9:0.1. Thus the rmse was lowered to was 0" + str(
        rootmeansqerr))
f1.write("\n")
f1.write("\n")
f1.write("Error Distribution:")
f1.write("\n")
f1.write(">=0 and <1: " + str(resdiff[0][1]))
f1.write("\n")
f1.write(">=1 and <2: " + str(resdiff[1][1]))
f1.write("\n")
f1.write(">=2 and <3: " + str(resdiff[2][1]))
f1.write("\n")
f1.write(">=3 and <4: " + str(resdiff[3][1]))
f1.write("\n")
f1.write(">=4: " + str(resdiff[4][1]))
f1.write("\n")
f1.write("\n")
f1.write("RSME:")
f1.write("\n")
f1.write(str(rootmeansqerr))
f1.write("\n")
f1.write("\n")
f1.write("Execution Time:")
f1.write("\n")
f1.write(str(end - start) + "s")


