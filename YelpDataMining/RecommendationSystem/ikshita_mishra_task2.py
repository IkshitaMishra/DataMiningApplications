import random
import sys

import math

from pyspark import SparkContext, SparkConf
import time
from pyspark.mllib.recommendation import ALS, Rating

sc = SparkContext()
sc.setLogLevel("WARN")

start = time.time()

#Read Data, Remove Headers  : training Data
csvrdd = sc.textFile(sys.argv[1])
head2 = csvrdd.first()
csvrdd = csvrdd.filter(lambda line:line!=head2).map(lambda x: x.split(","))

#Read Data, Remove Headers  : test Data
testrdd = sc.textFile(sys.argv[2])
head1 = testrdd.first()
testrdd = testrdd.filter(lambda line:line!=head1).map(lambda x: x.split(","))


if (int(sys.argv[3]) == 1):

	user = csvrdd.map(lambda a: (a[0], a[1])).reduceByKey(lambda a, b: a).keys()
	testuser = testrdd.map(lambda a: (a[0], a[1])).reduceByKey(lambda a, b: a).keys()

	buss = csvrdd.map(lambda a: (a[1], a[0])).reduceByKey(lambda a, b: a).keys()
	testbuss = testrdd.map(lambda a: (a[1], a[0])).reduceByKey(lambda a, b: a).keys()

	combusers = user.union(testuser).distinct().collect()
	combbuss = buss.union(testbuss).distinct().collect()

	# Map user id string with number : CSVDATA
	userdict = {}
	for i, j in enumerate(sorted(combusers)):
		userdict[j] = i

	# Map user id string with number
	bussdict = {}
	for i, j in enumerate(sorted(combbuss)):
		bussdict[j] = i

	vu = sc.broadcast(userdict)
	vb = sc.broadcast(bussdict)

	inv_u = {v: k for k, v in userdict.items()}
	inv_b = {v: k for k, v in bussdict.items()}

	testdata = testrdd.map(lambda x: ((int(vu.value[x[0]]), int(vb.value[x[1]])), float(x[2])))  # Initialize test value as 1
	csvdata = csvrdd.map(lambda x: ((int(vu.value[x[0]]), int(vb.value[x[1]])), float(x[2])))
	test = testdata.map(lambda x: x[0])
	trainingset = csvdata.subtractByKey(testdata)
	training = trainingset.map(lambda x:Rating(int(x[0][0]), int(x[0][1]), float(x[1])))
	# ================ Model Building =================================================
	modelCF = ALS.train(training,rank=5,iterations=10,lambda_=0.1)
	# ================ Predictions  ==================================================
	preds = modelCF.predictAll(test).map(lambda x: ((x[0], x[1]), x[2]))
	# ====================Cold Start Problem=====================================
	meanofall = preds.map(lambda x:x[1]).mean()
	coldstart = testdata.subtractByKey(preds).map(lambda x:((x[0][0],x[0][1]),float(meanofall)))
	predictions = preds.union(coldstart)
	# ================ Writing Output ===================================================
	f = open(sys.argv[4], "w")
	f.write("user_id, business_id, prediction")
	f.write("\n")
	predList = predictions.collect()
	for i in predList:
		f.write(inv_u[i[0][0]] + "," + inv_b[i[0][1]] + "," + str(i[1]))
		f.write("\n")

elif (int(sys.argv[3]) == 2):
	#No Mapping is required!
	testdata = testrdd.map(lambda x: ((x[0], x[1]), float(x[2])))  # Initialize test value as 1
	csvdata = csvrdd.map(lambda x: ((x[0],x[1]), float(x[2])))
	test = testdata.map(lambda x: x[0])
	trainingset = csvdata.subtractByKey(testdata)
	# (user,bus) : rating
	ubrating_dict = {}
	ubrating = trainingset.collect()
	#print(csvdata.take(10))
	#print(testdata.take(10))
	for i in ubrating:
		ubrating_dict[i[0]] = i[1]
	#print(ubrating_dict)
	# user : (bus, rating)
	user_dict = {}
	user = trainingset.map(lambda x: (x[0][0], (x[0][1], x[1]))).groupByKey().mapValues(list).collect()
	for i in user:
		user_dict[i[0]] = sorted(i[1])
	# user : (bus)
	user_bus_dict = {}
	for i in user:
		user_bus_dict[i[0]] = [x[0] for x in i[1]]
	# bus : (user,rating)
	buss_dict = {}
	buss = trainingset.map(lambda x: (x[0][1], (x[0][0],x[1]))).groupByKey().mapValues(list).collect()
	for i in buss:
		buss_dict[i[0]] = sorted(i[1])
	# user : avg(rating)
	user_avg_dict = {}
	for i in user:
		user_avg_dict[i[0]] = (sum([x[1] for x in i[1]]) / len(i[1]))

	def getSimandPredictUB(x):

		currUser = x[0]
		currBuss = x[1]
		simUserLi = list()
		if currBuss  in buss_dict:
			partnerUsers = []
			for i in buss_dict[currBuss]:
				partnerUsers.append(i[0])
			for partner in partnerUsers:
				if currUser != partner:
					N=13
					if len(set(user_bus_dict[currUser]).intersection(set(user_bus_dict[partner]))) >= N:

						# ============= Pearson Correlation (Similarity)===============

						currHistory = user_dict[currUser]
						partnerHistory = user_dict[partner]
						bothRated = list()
						currpointer = 0
						partnerpointer = 0
						clen = len(currHistory)
						plen = len(partnerHistory)
						# Create Matrix User - Bussines as in PPT ===============================
						while ( partnerpointer < plen and currpointer < clen):
							currRatedBuss = currHistory[currpointer][0]
							currRating = currHistory[currpointer][1]
							partRatedBuss= partnerHistory[partnerpointer][0]
							partRating= partnerHistory[partnerpointer][1]
							if currRatedBuss < partRatedBuss:
								currpointer +=1
							elif currRatedBuss > partRatedBuss:
								partnerpointer+=1
							elif currRatedBuss == partRatedBuss:
								bothRated.append(((currRating, partRating),0))
								partnerpointer += 1
								currpointer += 1
						if len(bothRated) == 0 or len(bothRated) == 1:
							sim= None
						else:
							denoA = 0.0
							numerator = 0.0
							denoB = 0.0
							curr_list = []
							part_list = []
							for x in bothRated:
								curr_list.append(x[0][0])
								part_list.append(x[0][1])
							for i in range(len(bothRated)):
								c = curr_list[i] - (sum(curr_list) / len(curr_list))
								p = part_list[i] - (sum(part_list) / len(part_list))
								numerator = numerator + (c * p)
								denoA = denoA + (c * c)
								denoB = denoB + (p * p)
							denomi = pow(denoA, 0.5) * pow(denoB, 0.5)
							if  numerator ==0:
								sim = None
							elif denomi == 0:
								sim= None
							else:
								sim = numerator / denomi

						if sim != None:
							simUserLi.append((partner,sim))
		elif currBuss not  in buss_dict:
			simavg = 3.0
			simUserLi.append((currUser,simavg))
		# =============== Predicting Rating =====================
		SIMLENG = len(simUserLi)
		if SIMLENG == 0:
			return ((currUser, currBuss), float(user_avg_dict[currUser]))
		if SIMLENG >= 1:
			#print("yas")
			numerator2 = 0
			denomi2 = 0
			for simU in simUserLi:
				partSimuser = simU[0]
				weight = simU[1]
				if (partSimuser, currBuss) in ubrating_dict:
					# check condition
					partToBussRate = ubrating_dict[(partSimuser, currBuss)]
					partSimuserHistory = user_dict[partSimuser]
					partSimuserRatings = []
					for x in partSimuserHistory:
						if partSimuserHistory[0] != currBuss:
							partSimuserRatings.append(x[1])
					avgPartRating = sum(partSimuserRatings) / len(partSimuserRatings)
					denomi2 = denomi2 + abs(weight)
					numerator2 = numerator2 + weight * (partToBussRate - avgPartRating)
			if denomi2 == 0:
				return ((currUser,currBuss),float(user_avg_dict[currUser]))
			else:
				ans = float(numerator2 / denomi2)
				return ((currUser,currBuss),float(user_avg_dict[currUser] + ans))

	predictions = test.map(lambda x:  getSimandPredictUB(x))
	#print(predictions.take(10))
	predList = predictions.collect()

	#print("predicted!")
	f = open(sys.argv[4], "w")
	f.write("user_id, business_id, prediction")
	f.write("\n")
	for i in predList:
		f.write(i[0][0] + "," + i[0][1] + "," + str(i[1]))
		f.write("\n")

elif (int(sys.argv[3]) == 3):
	# No Mapping is required!
	testdata = testrdd.map(lambda x: ((x[0], x[1]), float(x[2])))  # Initialize test value as 1
	csvdata = csvrdd.map(lambda x: ((x[0], x[1]), float(x[2])))
	test = testdata.map(lambda x: x[0])
	trainingset = csvdata.subtractByKey(testdata)
	# (user,bus) : rating
	ubrating_dict = {}
	ubrating = trainingset.collect()
	# print(csvdata.take(10))
	# print(testdata.take(10))
	for i in ubrating:
		ubrating_dict[i[0]] = i[1]
	# print(ubrating_dict)

	# user : (bus, rating)
	user_dict = {}
	user = trainingset.map(lambda x: (x[0][0], (x[0][1], x[1]))).groupByKey().mapValues(list).collect()
	for i in user:
		user_dict[i[0]] = sorted(i[1])

	# bus : (user,rating)
	buss_dict = {}
	buss = trainingset.map(lambda x: (x[0][1], (x[0][0], x[1]))).groupByKey().mapValues(list).collect()
	for i in buss:
		buss_dict[i[0]] = sorted(i[1])
	buss_user_dict ={}
	for i in buss:
		buss_user_dict[i[0]] = [x[0] for x in i[1]]

	# user : avg(rating)
	user_avg_dict = {}
	for i in user:
		user_avg_dict[i[0]] = (sum([x[1] for x in i[1]]) / len(i[1]))

	def genSimilarityAndPredictIB(x):

		currUser = x[0]
		currBuss = x[1]

		simBussLi = list()
		if currBuss  in buss_dict:
			partnerBuss = []
			for x in user_dict[currUser]:
				partnerBuss.append(x[0])
			for partner in partnerBuss:
				if currBuss != partner:
					N =25
					if len(set(buss_user_dict[currBuss]).intersection(set(buss_user_dict[partner]))) >= N:

						#============= Pearson Correlation (Similarity)===============
						currBussHistory = buss_dict[currBuss]
						partnerBussHistory = buss_dict[partner]
						bothRated = list()
						currpointer = 0
						partnerpointer = 0
						clen = len(currBussHistory)
						plen = len(partnerBussHistory)
						# Create Matrix User - Bussines as in PPT ===============================
						while ( partnerpointer < plen and currpointer < clen):
							currRatedBuss = currBussHistory[currpointer][0]
							currRating = currBussHistory[currpointer][1]
							partRatedBuss= partnerBussHistory[partnerpointer][0]
							partRating= partnerBussHistory[partnerpointer][1]
							if currRatedBuss < partRatedBuss:
								currpointer +=1
							elif currRatedBuss > partRatedBuss:
								partnerpointer+=1
							elif currRatedBuss == partRatedBuss:
								bothRated.append(((currRating, partRating),0))
								partnerpointer += 1
								currpointer += 1
						if len(bothRated) == 0 or len(bothRated) == 1:
							sim= None
						else:
							denoA = 0.0
							numerator = 0.0
							denoB = 0.0
							curr_list = []
							part_list = []
							for x in bothRated:
								curr_list.append(x[0][0])
								part_list.append(x[0][1])
							for i in range(len(bothRated)):
								c = curr_list[i] - (sum(curr_list) / len(curr_list))
								p = part_list[i] - (sum(part_list) / len(part_list))
								numerator = numerator + (c * p)
								denoA = denoA + (c * c)
								denoB = denoB + (p * p)
							denomi = pow(denoA, 0.5) * pow(denoB, 0.5)
							if  numerator ==0:
								sim = None
							elif denomi == 0:
								sim= None
							else:
								sim = numerator / denomi

						if sim != None:
							simBussLi.append((partner,sim))
		elif currBuss not in buss_dict:
			simavg = 3.0
			simBussLi.append((currBuss,simavg))
		# =============== Predicting Rating =====================
		SIMLENG = len(simBussLi)
		if SIMLENG == 0:
			return ((currUser, currBuss), float(user_avg_dict[currUser]))
		if SIMLENG >= 1:
			#print("yas")
			numerator2 = 0
			denomi2 = 0
			for simBuss in simBussLi:
				weight = simBuss[1]
				partSimBuss = simBuss[0]
				if (currUser, partSimBuss) in ubrating_dict:
					# check condition
					rating = ubrating_dict[(currUser, partSimBuss)]
					denomi2 = denomi2 + abs(weight)
					numerator2 = numerator2 + (weight * rating)
			if denomi2 == 0:
				return ((currUser,currBuss),float(user_avg_dict[currUser]))
			else:
				ans = abs(numerator2 / denomi2)
				return ((currUser,currBuss), ans)

	predictions = test.map(lambda x:  genSimilarityAndPredictIB(x))
	#print(predictions.take(10))
	predList = predictions.collect()

	#print("predicted!")
	f = open(sys.argv[4], "w")
	f.write("user_id, business_id, prediction")
	f.write("\n")
	for i in predList:
		f.write(i[0][0] + "," + i[0][1] + "," + str(i[1]))
		f.write("\n")


elif (int(sys.argv[3]) == 4):
	# No Mapping is required!
	testdata = testrdd.map(lambda x: ((x[0], x[1]), float(x[2])))  # Initialize test value as 1
	csvdata = csvrdd.map(lambda x: ((x[0], x[1]), float(x[2])))
	test = testdata.map(lambda x: x[0])
	trainingset = csvdata.subtractByKey(testdata)
	# Map user id string with number
	userdict = {}
	user1 = trainingset.map(lambda a: (a[0][0], a[0][1])).reduceByKey(lambda a, b: a).keys().collect()
	for i, j in enumerate(sorted(user1)):
		userdict[j] = i

	# print(userdict)

	# Groupbykey Business
	busssimdata = trainingset.map(lambda x: (x[0][1], userdict[x[0][0]])).groupByKey().mapValues(set).sortBy(
		lambda x: x[0])
	# print(busssimdata.take(100))
	bussusermappings = busssimdata.collect()
	m = len(userdict)
	# print(busssimdata.take(10))
	#print("m", m)
	bususermap = {}
	for i in bussusermappings:
		bususermap[i[0]] = i[1]

	def minhash(userids):
		minhas = [userids[0]]
		finalhash = []
		hashfunc = \
			[[87, 91, 671], [123, 192, 671], [35, 50, 671], [195, 164, 671], [32, 37, 671], [43, 21, 671],
			 [51, 58, 671],
			 [68, 73, 671],
			 [42, 78, 671], [136, 172, 671], [13, 19, 671], [93, 32, 671], [85, 95, 671], [19, 23, 671], [20, 27, 671],
			 [10, 29, 671],
			 [387, 552, 671], [11, 13, 671], [17, 29, 671], [53, 34, 671], [17, 35, 671], [77, 44, 671], [14, 93, 671],
			 [40, 82, 671],
			 [62, 87, 671], [73, 23, 671], [97, 5, 671], [17, 54, 671], [14, 83, 671], [130, 120, 671]]
		for each in hashfunc:
			selectmin = []
			for user in userids[1]:
				selectmin.append(((each[0] * user + each[1]) % each[2]) % m)
			finalhash.append(min(selectmin))
		minhas.append(finalhash)
		# print(minhas)
		return minhas


	sigmatrix = busssimdata.map(lambda a: (minhash(a)))
	bands = 15


	def createbands(tups):
		numhash = len(tups[1])
		rows = int(numhash / bands)
		return [(((tups[1][i], tups[1][i + 1]), i), tups[0]) for i in range(0, bands, rows)]


	bandData = sigmatrix.flatMap(lambda x: createbands(x)).groupByKey().mapValues(lambda x: sorted(list(x))).filter(
		lambda c: len(c[1]) >= 2)
	# print(bandData.take(10))
	# Creating Pairs
	#print("bands created")


	def createCandidates(busList):
		return {((busList[1][i], busList[1][j]), 1) for i in range(len(busList[1])) for j in
				range(i + 1, len(busList[1]))}


	candPairs = bandData.flatMap(lambda x: createCandidates(x)).reduceByKey(lambda c, d: c + d).keys()
	# .groupByKey().sortBy(lambda x:x[0])
	# print("cand",candPairs.take(10))
	#print("cands created")
	#print(candPairs.count())


	def jaccardsim(candPairs):
		intersect = (bususermap[candPairs[0]]).intersection(bususermap[candPairs[1]])
		uni = (bususermap[candPairs[0]]).union(bususermap[candPairs[1]])
		return len(intersect) / len(uni)

	simbus = candPairs.map(lambda x: (x[0], x[1], jaccardsim(x))).filter(lambda a: a[2] >= 0.4).map(lambda x: (x[0], x[1]))

	setSimPairsA = simbus.map(lambda x: (x[0], x[1])).groupByKey().mapValues(list).collect()
	setSimPairsB = simbus.map(lambda x: (x[1], x[0])).groupByKey().mapValues(list).collect()

	simPairA_Dict = {}
	for i in setSimPairsA:
		simPairA_Dict[i[0]] = i[1]
	simPairB_Dict = {}
	for i in setSimPairsB:
		simPairB_Dict[i[0]] = i[1]
	#print(simPairB_Dict)
	#print(simPairA_Dict)

	# (user,bus) : rating
	ubrating_dict = {}
	ubrating = trainingset.collect()
	# print(csvdata.take(10))
	# print(testdata.take(10))
	for i in ubrating:
		ubrating_dict[i[0]] = i[1]
	# print(ubrating_dict)

	# user : (bus, rating)
	user_dict = {}
	user = trainingset.map(lambda x: (x[0][0], (x[0][1], x[1]))).groupByKey().mapValues(list).collect()
	for i in user:
		user_dict[i[0]] = sorted(i[1])

	# bus : (user,rating)
	buss_dict = {}
	buss = trainingset.map(lambda x: (x[0][1], (x[0][0], x[1]))).groupByKey().mapValues(list).collect()
	for i in buss:
		buss_dict[i[0]] = sorted(i[1])

	buss_user_dict ={}
	for i in buss:
		buss_user_dict[i[0]] = [x[0] for x in i[1]]
	# user : avg(rating)
	user_avg_dict = {}
	for i in user:
		user_avg_dict[i[0]] = (sum([x[1] for x in i[1]]) / len(i[1]))


	def genSimilarityAndPredictIBwithLSH(x):
		currUser = x[0]
		currBuss = x[1]
		simBussLi = list()
		if (currBuss not in simPairA_Dict) and (currBuss not in simPairB_Dict):
			simavg = 3.0
			simBussLi.append((currBuss, simavg))
		elif (currBuss not in buss_dict):
			simavg = 3.0
			simBussLi.append((currBuss, simavg))
		else:
			partnerBuss = []
			if currBuss in simPairA_Dict:
				partnerBuss.extend(simPairA_Dict[currBuss])
			if currBuss in simPairB_Dict:
				partnerBuss.extend(simPairB_Dict[currBuss])
			partnerBuss = list(set(partnerBuss))
			#print(partnerBuss)
			for partner in partnerBuss:
				if currBuss != partner:
					# ============= Pearson Correlation (Similarity)===============
					currBussHistory = buss_dict[currBuss]
					partnerBussHistory = buss_dict[partner]
					bothRated = list()
					currpointer = 0
					partnerpointer = 0
					clen = len(currBussHistory)
					plen = len(partnerBussHistory)
					# Create Matrix User - Bussines as in PPT ===============================
					while ( partnerpointer < plen and currpointer < clen):
						currRatedBuss = currBussHistory[currpointer][0]
						currRating = currBussHistory[currpointer][1]
						partRatedBuss= partnerBussHistory[partnerpointer][0]
						partRating= partnerBussHistory[partnerpointer][1]
						if currRatedBuss < partRatedBuss:
							currpointer +=1
						elif currRatedBuss > partRatedBuss:
							partnerpointer+=1
						elif currRatedBuss == partRatedBuss:
							bothRated.append(((currRating, partRating),0))
							partnerpointer += 1
							currpointer += 1
					if len(bothRated) == 0 or len(bothRated) == 1:
						sim= None
					else:

						# Computation =====
						denoA = 0.0
						numerator = 0.0
						denoB = 0.0
						curr_list = []
						part_list = []
						for x in bothRated:
							curr_list.append(x[0][0])
							part_list.append(x[0][1])
						for i in range(len(bothRated)):
							c = curr_list[i] - (sum(curr_list) / len(curr_list))
							p = part_list[i] - (sum(part_list) / len(part_list))
							numerator = numerator + (c * p)
							denoA = denoA + (c * c)
							denoB = denoB + (p * p)
						denomi = pow(denoA, 0.5) * pow(denoB, 0.5)
						if  numerator ==0:
							sim = None
						elif denomi == 0:
							sim= None
						else:
							sim = numerator / denomi
							#print(partner,sim)

					if sim != None:
						simBussLi.append((partner,sim))
		#print(simBussLi)
		#print(simBussLi)
		# =============== Predicting Rating =====================
		SIMLENG = len(simBussLi)
		if SIMLENG == 0:
			return ((currUser, currBuss), float(user_avg_dict[currUser]))
		if SIMLENG >= 1:
			#print("yas")
			numerator2 = 0
			denomi2 = 0
			for simBuss in simBussLi:
				weight = simBuss[1]
				partSimBuss = simBuss[0]
				if (currUser, partSimBuss) in ubrating_dict:
					# check condition
					rating = ubrating_dict[(currUser, partSimBuss)]
					denomi2 = denomi2 + abs(weight)
					numerator2 = numerator2 + (weight * rating)
			if denomi2 == 0:
				return ((currUser,currBuss),float(user_avg_dict[currUser]))
			else:
				#print("entered here as well!")
				ans = abs(numerator2 / denomi2)
				#print((currUser,currBuss))
				#print(ans)
				return ((currUser,currBuss), ans)

	
	predictions = test.map(lambda x:  genSimilarityAndPredictIBwithLSH(x))
	#print(predictions.take(10))
	predList = predictions.collect()


	#print("predicted!")
	f = open(sys.argv[4], "w")
	f.write("user_id, business_id, prediction")
	f.write("\n")
	for i in predList:
		f.write(i[0][0] + "," + i[0][1] + "," + str(i[1]))
		f.write("\n")

	f1 = open("ikshita_mishra_explanation.txt", "w")
	f1.write("Item-based CF recommendation system with Jaccard based LSH is better than : Item-based CF recommendation system because, using Locality Sensitive Hashing (Case 4) we get a set of similar businesses. Hence we calculate weights (Pearson Correlation) on only those similar pairs. In Case 3, we compute Pearson Correlation in all pairs. This takes time. Hence Item Based with LSH for finding similar pairs, is much faster and better. This result decrease in computation")


# ================ Count of Absolute Difference in each levels =======================

res = predictions.join(testdata)
absdiff = res.map(lambda x: abs(x[1][0]-x[1][1]))
def level(absdif):
	if(absdif>=0 and absdif<1):
		return (0,1)
	elif(absdif>=1 and absdif<2):
		return (1,1)
	elif(absdif>=2 and absdif<3):
		return (2,1)
	elif(absdif>=3 and absdif<4):
		return (3,1)
	elif(absdif>=4):
		return (4,1)
resdiff = absdiff.map(lambda x : level(x)).reduceByKey(lambda x, y : x+y).sortByKey().collect()

#print(resdiff)
print(">=0 and <1: "+ str(resdiff[0][1]))
print(">=1 and <2: "+ str(resdiff[1][1]))
print(">=2 and <3: " + str(resdiff[2][1]))
print(">=3 and <4: "+ str(resdiff[3][1]))
print(">=4: "+str(resdiff[4][1]))

# ================ Root Mean Squared Error =========================================

rootmeansqerr = pow(absdiff.map(lambda x: x*x).mean(), 0.5)
print(absdiff.map(lambda x: x*x).mean())
print("RMSE", rootmeansqerr)


end = time.time()
print("Duration", end-start)
