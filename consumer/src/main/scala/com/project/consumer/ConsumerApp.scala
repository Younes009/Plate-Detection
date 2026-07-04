package com.project.consumer

import com.project.common._
import com.project.ml.ModelLoader
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.streaming.Trigger
import org.apache.spark.sql.types.StringType

import scala.util.Try

/**
 * Microservice 2 — Consumer STRUCTURED STREAMING (Spark Structured Streaming + Scala).
 *
 * Args : [destPath] [outputPath] [modelPath] [checkpointPath]
 *
 * Logique :
 *   1. Charge le modèle ML via ModelLoader.initialize()
 *   2. Lit en streaming depuis destPath/ (Parquet brut produit par ProducerApp)
 *   3. Pour chaque micro-batch (foreachBatch) :
 *        a. Joint les labels YOLO : image_path → label_path + label_data
 *        b. Score : prediction + confidence_score
 *        c. Écriture 1 : flux complet → outputPath/
 *        d. Écriture 2 : prédictions → outputPath/predictions/
 *   4. awaitTermination() — tourne indéfiniment (exactly-once via checkpoint)
 */
object ConsumerApp {

  // image URI (file:///path/images/foo.jpg) → label URI (file:///path/labels/foo.txt)
  private val toLabelUri = udf { imgUri: String =>
    if (imgUri == null) null
    else imgUri
      .replaceAll("(?i)/images/", "/labels/")
      .replaceAll("(?i)\\.(jpg|jpeg|png)$", ".txt")
  }

  // Lit le contenu du fichier label depuis son URI file:/// — null si absent
  private val readLabel = udf { labelUri: String =>
    if (labelUri == null) null
    else Try {
      val file = new java.io.File(new java.net.URI(labelUri))
      if (file.exists()) new String(java.nio.file.Files.readAllBytes(file.toPath), "UTF-8")
      else null
    }.getOrElse(null): String
  }

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

    val query = rawStream.writeStream
      .foreachBatch { (batchDF: DataFrame, batchId: Long) =>
        if (!batchDF.isEmpty) {
          // Jointure label : dérive label_path depuis image_path, lit contenu YOLO
          val withLabels = batchDF
            .withColumn("label_path", toLabelUri(col("image_path")))
            .withColumn("label_data",  readLabel(col("image_path")))
            .withColumn("processed_at", current_timestamp())

          val scored = ModelLoader.score(withLabels)

          val outputCols = Schemas.outputSchema.fieldNames
          val toWrite    = scored.select(outputCols.filter(scored.columns.contains).map(col): _*)

          toWrite.write
            .mode("append")
            .option("compression", "snappy")
            .parquet(outputPath)

          toWrite.select("image_path", "label_data", "prediction", "confidence_score", "processed_at")
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
