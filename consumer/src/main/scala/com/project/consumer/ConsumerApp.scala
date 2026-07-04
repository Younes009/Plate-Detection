package com.project.consumer

import com.project.common._
import com.project.ml.ModelLoader
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.streaming.Trigger

/**
 * Microservice 2 — Consumer STRUCTURED STREAMING (Spark Structured Streaming + Scala).
 *
 * Args : [destPath] [outputPath] [modelPath] [checkpointPath]
 *
 * Logique :
 *   1. Charge le modèle ML (ONNX → MLlib → simulé) via ModelLoader.initialize()
 *   2. Lit en streaming depuis destPath/ (Parquet produit par ProducerApp, schéma explicite)
 *   3. Pour chaque micro-batch (foreachBatch) :
 *        a. Score les images : prediction + confidence_score
 *        b. Écriture 1 : flux complet → outputPath/          (append, Parquet Snappy)
 *        c. Écriture 2 : prédictions → outputPath/predictions/ (append, Parquet Snappy)
 *   4. awaitTermination() — tourne indéfiniment (exactly-once via checkpoint)
 */
object ConsumerApp {

  def main(args: Array[String]): Unit = {
    val destPath       = if (args.length > 0) args(0) else "data/dest"
    val outputPath     = if (args.length > 1) args(1) else "data/output"
    val modelPath      = if (args.length > 2) args(2) else "data/models"
    val checkpointPath = if (args.length > 3) args(3) else "data/checkpoint"

    val outputPredictions = outputPath.stripSuffix("/").stripSuffix("\\") +
                            java.io.File.separator + "predictions" + java.io.File.separator

    val spark = SparkSession.builder()
      .appName("PlateDetection-Consumer-Streaming")
      .getOrCreate()

    val modelPathNorm = modelPath.stripSuffix("/").stripSuffix("\\") + java.io.File.separator
    ModelLoader.initialize(spark, modelPathNorm)

    // recursiveFileLookup=true : bypass partition discovery sur batch_N_TS/
    val rawStream = spark.readStream
      .schema(Schemas.destSchema)
      .option("maxFilesPerTrigger", 1)
      .option("recursiveFileLookup", "true")
      .parquet(destPath)

    val enriched = rawStream.withColumn("processed_at", current_timestamp())

    val query = enriched.writeStream
      .foreachBatch { (batchDF: DataFrame, batchId: Long) =>
        if (!batchDF.isEmpty) {
          val scored = ModelLoader.score(batchDF)

          val outputCols = Schemas.outputSchema.fieldNames
          val toWrite    = scored.select(outputCols.filter(scored.columns.contains).map(col): _*)

          toWrite.write
            .mode("append")
            .option("compression", "snappy")
            .parquet(outputPath)

          // Prédictions + ground truth YOLO pour évaluation
          toWrite.select("image_path", "prediction", "confidence_score", "processed_at", "label_data")
            .write
            .mode("append")
            .option("compression", "snappy")
            .parquet(outputPredictions)

          println(s"[Consumer] batch $batchId : ${batchDF.count()} lignes traitées")
        }
      }
      .option("checkpointLocation",
        checkpointPath.stripSuffix("/").stripSuffix("\\") +
        java.io.File.separator + "consumer" + java.io.File.separator)
      .trigger(Trigger.ProcessingTime("10 seconds"))
      .start()

    query.awaitTermination()
  }
}
