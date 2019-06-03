import tweepy
from tweepy import OAuthHandler, StreamListener
from random import *
from pyspark import SparkContext


sc = SparkContext(appName="Twitter Streaming")
sc.setLogLevel("ERROR")
sc.setLogLevel(logLevel="OFF")

cK = ""
cS = ""
aK = ""
aS = ""

auth = tweepy.OAuthHandler(cK, cS)
auth.set_access_token(aK, aS)
api = tweepy.API(auth)

#Check whether api working
#user = api.me()
#print(user.name)

curr = 0
sampleList = []
maxLim =100
tagDict ={}
class streamLi(StreamListener):

    def on_status(self, status):
        #print(status.entities.get('hashtags'))
        global sampleList
        global curr
        global tagDict
        tags = status.entities.get('hashtags')
        hashLen = len(tags)
        if hashLen != 0:
            curr = curr + 1
            if curr <= maxLim:
                sampleList.append(status.text)
                for i in tags:
                    if i['text'] not in tagDict:
                        tagDict[i['text']] = 1
                    else:
                        tagDict[i['text']] = tagDict[i['text']] + 1
            elif curr > maxLim:
                randInteger = randint(1, curr)
                if maxLim > randInteger:
                    sampleList[randInteger] = status.text
                    for i in tags:
                        if i['text'] not in tagDict:
                            tagDict[i['text']] = 1
                        else:
                            tagDict[i['text']] = tagDict[i['text']] + 1

            print("\n")
            print("The number of tweets with tags from beginning: " + str(curr))
            vals = sorted(set(tagDict.values()), reverse=True)[:3]
            for i in vals:
                j = sorted([k for k,v in tagDict.items() if v == i])
                for m in j:
                    print(m + " : " + str(i))


    def on_error(self, status_code):
        if status_code == 420:
            return False

myStream = tweepy.Stream(auth = api.auth, listener=streamLi())
myStream.filter(track=["Paris"])


