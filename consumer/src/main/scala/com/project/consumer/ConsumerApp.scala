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
 *        a. Filtre les anomalies (num_boxes, bbox_area_norm, aspect_ratio)
 *        b. Enrichissement : quality_flag + processed_at
 *        c. Scoring ML : prediction + confidence_score
 *        d. Sélection du schéma de sortie (outputSchema)
 *        e. Écriture 1 : flux complet → outputPath/          (append, Parquet Snappy)
 *        f. Écriture 2 : prédictions → outputPath/predictions/ (append, Parquet Snappy)
 *   4. awaitTermination() — tourne indéfiniment (exactly-once via checkpoint)
 */
object ConsumerApp {

  def main(args: Array[String]): Unit = {
    val destPath       = if (args.length > 0) args(0) else HdfsPaths.DEST
    val outputPath     = if (args.length > 1) args(1) else HdfsPaths.OUTPUT
    val modelPath      = if (args.length > 2) args(2) else HdfsPaths.MODELS
    val checkpointPath = if (args.length > 3) args(3) else HdfsPaths.CHECKPOINT

    val outputPredictions = outputPath.stripSuffix("/") + "/predictions/"

    val spark = SparkSession.builder()
      .appName("PlateDetection-Consumer-Streaming")
      .getOrCreate()

    // Chargement du modèle ML (une seule fois, depuis le driver)
    val modelPathNorm = modelPath.stripSuffix("/").stripSuffix("\\") + java.io.File.separator
    ModelLoader.initialize(spark, modelPathNorm)

    // --- Lecture en streaming depuis DEST/ avec schéma explicite ---
    // recursiveFileLookup=true bypasses Hive partition discovery on batch_N_TS/ dirs
    val rawStream = spark.readStream
      .schema(Schemas.destSchema)
      .option("maxFilesPerTrigger", 1)
      .option("recursiveFileLookup", "true")
      .parquet(destPath)

    // --- Enrichissement (filtre + quality_flag + processed_at) ---
    val enriched = rawStream
      .filter(col("num_boxes").isNotNull)
      .withColumn("quality_flag",
        when(col("bbox_area_norm").isNull || col("num_boxes") === 0, lit("NO_PLATE"))
          .when(col("bbox_area_norm") < 0.005,                       lit("TOO_SMALL"))
          .when(col("aspect_ratio") > 6 || col("aspect_ratio") < (1.0 / 6), lit("BAD_RATIO"))
          .otherwise(lit("OK")))
      .withColumn("processed_at", current_timestamp())

    // --- foreachBatch : scoring + double écriture ---
    val query = enriched.writeStream
      .foreachBatch { (batchDF: DataFrame, batchId: Long) =>
        if (!batchDF.isEmpty) {
          val scored = ModelLoader.score(batchDF)

          // Sélection stricte selon outputSchema (colonnes dans l'ordre attendu)
          val outputCols = Schemas.outputSchema.fieldNames
          val toWrite    = scored.select(outputCols.filter(scored.columns.contains).map(col): _*)

          toWrite.write
            .mode("append")
            .option("compression", "snappy")
            .parquet(outputPath)

          toWrite.select("image_path", "prediction", "confidence_score", "processed_at")
            .write
            .mode("append")
            .option("compression", "snappy")
            .parquet(outputPredictions)

          println(s"[Consumer] batch $batchId : ${batchDF.count()} lignes traitées")
        }
      }
      .option("checkpointLocation", checkpointPath.stripSuffix("/").stripSuffix("\\") + java.io.File.separator + "consumer" + java.io.File.separator)
      .trigger(Trigger.ProcessingTime("10 seconds"))
      .start()

    query.awaitTermination()
  }
}
