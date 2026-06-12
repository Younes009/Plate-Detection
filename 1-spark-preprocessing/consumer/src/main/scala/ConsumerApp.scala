import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.streaming.Trigger

/**
 * Consumer STREAMING (Spark Structured Streaming + Scala).
 * - Lit en continu hdfs:///project/raw/ (readStream, Parquet, dataset prétraité plaques)
 * - Filtre les images sans détection + ajoute un flag qualité sur la bbox
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
      .schema(Schemas.preprocessedSchema)
      .parquet(HdfsPaths.RAW)

    // --- 2. Transformations : filtre + enrichissement ---
    val enrichedStream = rawStream
      .filter(col("num_boxes") > 0)
      .withColumn("quality_flag",
        when(col("bbox_area_norm") < 0.005, lit("TOO_SMALL"))
          .when(col("aspect_ratio") > 6 || col("aspect_ratio") < (1.0 / 6), lit("BAD_RATIO"))
          .otherwise(lit("OK")))
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
      .withWatermark("ingested_at", "1 minute")
      .groupBy(
        window(col("ingested_at"), "1 minute"),
        col("split"),
        col("bbox_size_category")
      )
      .agg(
        count("*").as("nb_images"),
        avg("aspect_ratio").as("avg_aspect_ratio"),
        avg("num_boxes").as("avg_num_boxes")
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
