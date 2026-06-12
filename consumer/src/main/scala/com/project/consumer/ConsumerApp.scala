package com.project.consumer

import com.project.common._
import com.project.ml.ModelLoader
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.streaming.Trigger

/**
 * Consumer STREAMING (Spark Structured Streaming + Scala).
 * - FLUX 1 (data) : readStream Parquet sur hdfs:///project/raw/ avec schéma explicite
 * - FLUX 2 (modèle ML) : ModelLoader (PipelineModel MLlib si disponible, sinon scoring simulé réaliste)
 * - Par micro-batch : filtre les anomalies, enrichit (features + quality_flag), score ML
 *   (`prediction`, `confidence_score`)
 * - Double écriture :
 *     1. flux enrichi complet -> hdfs:///project/processed/ (append)
 *     2. sous-ensemble (image_path, prediction, confidence_score) -> hdfs:///project/processed/predictions/ (append)
 * - Checkpointing strict sous hdfs:///project/checkpoint/ (exactly-once), tourne indéfiniment (awaitTermination)
 */
object ConsumerApp {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("PlateDetection-Consumer-Streaming")
      .getOrCreate()

    // --- FLUX 1 : lecture en streaming depuis HDFS (raw), schéma explicite ---
    val rawStream = spark.readStream
      .schema(Schemas.preprocessedSchema)
      .parquet(HdfsPaths.RAW)

    // --- Filtrage des anomalies + feature engineering de production ---
    val enriched = rawStream
      .filter(col("num_boxes") > 0)
      .withColumn("quality_flag",
        when(col("bbox_area_norm") < 0.005, lit("TOO_SMALL"))
          .when(col("aspect_ratio") > 6 || col("aspect_ratio") < (1.0 / 6), lit("BAD_RATIO"))
          .otherwise(lit("OK")))
      .withColumn("processed_at", current_timestamp())

    // --- FLUX 2 : scoring ML (modèle MLlib si dispo, sinon simulé réaliste) ---
    val scored = ModelLoader.score(enriched)

    // --- Écriture principale : flux enrichi -> processed/ ---
    val processedQuery = scored.writeStream
      .format("parquet")
      .option("path", HdfsPaths.PROCESSED)
      .option("checkpointLocation", HdfsPaths.CHECKPOINT + "processed/")
      .outputMode("append")
      .trigger(Trigger.ProcessingTime("10 seconds"))
      .start()

    // --- Écriture dédiée aux prédictions -> processed/predictions/ ---
    val predictionsQuery = scored
      .select(col("image_path"), col("prediction"), col("confidence_score"))
      .writeStream
      .format("parquet")
      .option("path", HdfsPaths.PROCESSED_PREDICTIONS)
      .option("checkpointLocation", HdfsPaths.CHECKPOINT + "predictions/")
      .outputMode("append")
      .trigger(Trigger.ProcessingTime("10 seconds"))
      .start()

    // --- Maintien en exécution indéfinie ---
    spark.streams.awaitAnyTermination()
  }
}
