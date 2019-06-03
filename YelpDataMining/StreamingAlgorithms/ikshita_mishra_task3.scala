

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import twitter4j.Status
import org.apache.spark.streaming.{Seconds, StreamingContext}
import scala.collection.mutable.ListBuffer
import scala.util.Random
import org.apache.spark.streaming.twitter._


object ikshita_mishra_task3 {

  var curr = 0
  var sampleList: ListBuffer[String] = scala.collection.mutable.ListBuffer.empty[String]
  var maxLim = 100
  var tagDict = scala.collection.mutable.HashMap.empty[String, Int]


  def task(rdd : RDD[Status]) : Unit = {

    val twt = rdd.collect()
    for(status <- twt){

      val tags = status.getHashtagEntities().map(_.getText)
      if (tags.length != 0)
      {
        curr += 1
        if (curr <= maxLim){
          sampleList.append(status.getText)
          for (i <- tags) {

            if (tagDict.contains(i))
            {
              tagDict(i) += 1
            } else
            {
              tagDict(i) = 1
            }
          }
        }
        else {
          val rand = Random.nextInt(curr)
          if (rand < maxLim){
            sampleList(rand) = status.getText
          }
          for (i <- tags) {
            if (tagDict.contains(i))
            {
              tagDict(i) += 1
            } else
            {
              tagDict(i) = 1
            }
          }
        }
        print("\n")
        def Desc[T : Ordering] = implicitly[Ordering[T]].reverse
        print("The number of tweets with tags from beginning: " + curr)
        print("\n")
        val values =  tagDict.values.toSet.toList.sorted(Ordering[Int].reverse).slice(0,3)
        for (b <- values) {
          val filtered = tagDict.filter((t) => t._2 == b)
          val fil = filtered.keys.toList.sorted
          for (ff<- fil) {
            print(ff + " : " + b)
            print("\n")
          }
        }
      }
    }
  }

  def main(args: Array[String]): Unit = {

    val spark = new SparkConf()
    spark.setAppName("Stream")

    spark.setMaster("local[*]")

    val sc = new SparkContext(spark)
    sc.setLogLevel(logLevel = "OFF")

    System.setProperty("twitter4j.oauth.consumerKey", "3y2bHoP2pfxZQBdSeupMJVfSf")
    System.setProperty("twitter4j.oauth.consumerSecret", "vjlrJ0hsMXl0QHiAPoVWETjWtfmCLiPAbxGwykiyZ4N35AE5xv")
    System.setProperty("twitter4j.oauth.accessToken", "1033449764795297794-oWXvn9Z00mxcEMVArRBGeAJDyaqRXa")
    System.setProperty("twitter4j.oauth.accessTokenSecret", "f41twxE2QhobND1VGaBTNIuBbkKQivQttaqa6NEczcyao")

    val ssc = new StreamingContext(sc, Seconds(1))
    val tw = TwitterUtils.createStream(ssc, None, Array("Paris"))

    tw.foreachRDD(rdd => task(rdd))


    ssc.start()
    ssc.awaitTermination()
  }
}