import org.apache.spark._
import org.apache.log4j.Logger
import org.apache.log4j.Level
import java.text.SimpleDateFormat
import java.util.{Date, TimeZone}
import org.json4s._
import java.io._
import breeze.linalg.max
import org.json4s.jackson.JsonMethods._
import org.apache.spark.streaming._

import scala.collection.mutable.ListBuffer

object ikshita_mishra_task2 {

  def main(args: Array[String]): Unit = {


    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)


    val conf = new SparkConf().setMaster("local[*]").setAppName("Flajolet")
    val ssc = new StreamingContext(conf, Seconds(5))
    val lines = ssc.socketTextStream("localhost",args(0).toInt)

    val words = lines.map(x => pretty(render(parse(x)\"city")))


    val pw = new PrintWriter(new File(args(1)))
    pw.write("Time,Gound Truth,Estimation")
    pw.write("\n")
    pw.close()


    words.window(Seconds(30),Seconds(10)).foreachRDD(rdd => {
      val cities = rdd.collect()
      //println(cities)
      var groundSet = scala.collection.mutable.Set[Int]()
      val lis = cities.toList
      var sToInt: ListBuffer[Int] = scala.collection.mutable.ListBuffer.empty[Int]
      for ( i <- lis){
        val hashVal: Int = i.hashCode
        sToInt.append(hashVal)
        groundSet.add(hashVal)
      }


      val hashFunc: List[List[Int]] =
        List(List(87, 91, 671), List(123, 192, 671), List(35, 50,671),
          List(195, 164, 671), List(136, 172, 671), List(13, 19,671),
          List(93, 32, 671), List(85, 85, 671), List(19, 23,671))


      val hashBit: Array[Int] = Array.fill(hashFunc.length)(0)

      for (m <- hashFunc){
        var minLi: ListBuffer[Int] = scala.collection.mutable.ListBuffer.empty[Int]
        val num = hashFunc.indexOf(m)
        for (n <- sToInt)
        {

          val hashVal = ((m(0) * n + m(1)) % m(2)) % sToInt.length
          val an = hashVal.toBinaryString
          def rtrim(s: String) = s.replaceAll("0+$", "")
          val  zero = an.length - rtrim(an).length
          if (zero!=an.length){
            minLi.append(zero)
          }
          else if (zero==an.length){
            minLi.append(0)
          }
        }
        hashBit(num) = max(minLi)
      }
      var a1 = 0
      val aLi = List.range(0, 3)
      for (i <- aLi) {
        a1 = a1 + ((hashBit(i)) * (hashBit(i)))
      }
      a1 = a1 / aLi.length

      var b1 = 0
      val bLi = List.range(3, 6)
      for (j <- bLi) {
        b1 = b1 + ((hashBit(j)) * (hashBit(j)))
      }
      b1 = b1 / bLi.length

      var c1 = 0
      val cLi = List.range(6, 9)
      for (k <- cLi) {
        c1 = c1 + ((hashBit(k)) * (hashBit(k)))
      }
      c1 = c1 / cLi.length

      val li = List(a1, b1, c1).sorted
      val predictedCount = (li(1)).toInt

      val dateFor = new SimpleDateFormat("yyyy-MM-dd hh:mm:ss")
      dateFor.setTimeZone(TimeZone.getTimeZone("America/Los_Angeles"))
      var subD = new Date()
      val dt = dateFor.format(subD)
      val fw = new FileWriter(args(1), true)
      fw.write(dt.toString() +","+ groundSet.size.toString()+","+ predictedCount.toString()+"\n")
      fw.close()

    })

    ssc.start()
    ssc.awaitTermination()

  }

}