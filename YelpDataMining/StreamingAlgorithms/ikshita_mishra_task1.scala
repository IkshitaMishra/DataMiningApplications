import breeze.numerics.abs
import org.apache.spark._
import java.text.SimpleDateFormat
import java.util.{Date, TimeZone}
import org.json4s._
import java.io._
import org.json4s.jackson.JsonMethods._
import scala.collection.mutable.ListBuffer
import org.apache.spark.streaming._
import org.apache.log4j.Logger
import org.apache.log4j.Level

object ikshita_mishra_task1 {

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val conf = new SparkConf().setMaster("local[*]").setAppName("Bloom")
    val ssc = new StreamingContext(conf, Seconds(10))
    val lines = ssc.socketTextStream("localhost", args(0).toInt)

    val words = lines.map(x => pretty(render(parse(x)\"city")))

    val bitArray: Array[Int] = Array.fill(200)(0)
    var instream = scala.collection.mutable.Set[Int]()
    var falsePos: Int = 0
    var trueNeg: Int = 0


    val pw = new PrintWriter(new File(args(1)))
    pw.write("Time,FPR")
    pw.write("\n")
    pw.close()
    words.foreachRDD(rdd => {
      var cities = rdd.collect()
      var lis = cities.toList
      var sToInt: ListBuffer[Int] = scala.collection.mutable.ListBuffer.empty[Int]
      for ( i <- lis){
        var hashVal: Int = i.hashCode
        sToInt.append(hashVal)
      }
      for (j <- sToInt){
        var h1 = abs((((87 * j) + 91) % 671) % bitArray.length)
        var h2 = abs((((93 * j) + 32) % 671) % bitArray.length)
        var h3 = abs((((85 * j) + 95) % 671) % bitArray.length)

        var a1 = bitArray(h1)
        var a2 = bitArray(h2)
        var a3 = bitArray(h3)

        if ((a1== 0) || (a2== 0) || (a3== 0) ) {
          if(!instream.contains(j))
          {
            trueNeg = trueNeg + 1
          }
          bitArray(h1) = 1
          bitArray(h2) = 1
          bitArray(h3) = 1
          instream.add(j)
        }
        else if ((a1== 1) && (a2== 1) && (a3== 1)) {
          if(!(instream.contains(j)))
          {
            falsePos = falsePos + 1
          }
          instream.add(j)
        }
      }
      val dateFor = new SimpleDateFormat("yyyy-MM-dd hh:mm:ss")
      dateFor.setTimeZone(TimeZone.getTimeZone("America/Los_Angeles"))
      var subD = new Date()
      val dt = dateFor.format(subD)
      if ( falsePos == 0 && trueNeg == 0)  {
        val fw = new FileWriter(args(1), true)
        fw.write(dt.toString() +","+ 0.0 +"\n")
        fw.close()
      }
      else {
        val fw = new FileWriter(args(1), true)
        fw.write(dt.toString() +"," + (falsePos.toFloat / (falsePos + trueNeg).toFloat) +"\n")
        fw.close()
      }
    })
    ssc.start()
    ssc.awaitTermination()
  }
}



