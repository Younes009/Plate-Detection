import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.streaming.Trigger

/**
 * Consumer STREAMING (Spark Structured Streaming + Scala).
 * - Lit en continu hdfs:///project/raw/ (readStream, Parquet)
 * - Filtre, enrichit (risk_score)
 * - Écrit en continu vers hdfs:///project/processed/
 * - Checkpoint : hdfs:///project/checkpoint/
 * - Tourne indéfiniment (awaitTermination)
 */
object ConsumerApp {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("PlateDetection-Consumer-Streaming")
      .getOrCreate()

    // --- 1. Lecture en streaming depuis HDFS (raw) ---
    val rawStream = spark.readStream
      .schema(Schemas.plateDetectionSchema)
      .parquet(HdfsPaths.RAW)

    // --- 2. Transformations : filtre + enrichissement ---
    val enrichedStream = rawStream
      .filter(col("confidence") >= 0.5)
      .withColumn("risk_level", Scoring.riskScoreExpr)
      .withColumn("processed_at", current_timestamp())

    // --- 3. Écriture en continu vers processed (Parquet + checkpoint) ---
    val processedQuery = enrichedStream.writeStream
      .format("parquet")
      .option("path", HdfsPaths.PROCESSED)
      .option("checkpointLocation", HdfsPaths.CHECKPOINT + "processed/")
      .outputMode("append")
      .trigger(Trigger.ProcessingTime("10 seconds"))
      .start()

    // --- 3bis. (Bonus) Agrégation streaming par fenêtre temporelle ---
    val aggStream = rawStream
      .withWatermark("detected_at", "1 minute")
      .groupBy(
        window(col("detected_at"), "1 minute"),
        col("vehicle_type")
      )
      .agg(
        count("*").as("nb_detections"),
        avg("confidence").as("avg_confidence")
      )

    val aggQuery = aggStream.writeStream
      .format("parquet")
      .option("path", HdfsPaths.PROCESSED + "aggregates/")
      .option("checkpointLocation", HdfsPaths.CHECKPOINT + "aggregates/")
      .outputMode("append")
      .trigger(Trigger.ProcessingTime("10 seconds"))
      .start()

    // --- 4. Maintien en exécution indéfinie ---
    spark.streams.awaitAnyTermination()
  }
}
