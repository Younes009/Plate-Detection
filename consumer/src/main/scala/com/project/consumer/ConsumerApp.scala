package com.project.consumer

import com.project.common.Schemas
import com.project.ml.ModelLoader
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.streaming.Trigger

/**
 * Microservice 2 — Consumer & Predictor (Spark Structured Streaming + Scala).
 *
 * Args : [ingressPath] [stagingPath] [outputPath] [modelPath] [checkpointPath]
 *
 * Tourne EN PERMANENCE (awaitTermination, checkpoint exactly-once) — c'est le
 * seul service permanent du pipeline. Structured Streaming natif : aucun
 * DStream, aucune boucle manuelle.
 *
 * Il écoute EXCLUSIVEMENT DEST/continuous_ingress/ : des images brutes, sans
 * aucun label YOLO (les labels n'existent que dans ml_dataset, consommé par le
 * service ML Python — jamais par le stream). Aucune recherche de fichier .txt.
 *
 * Ordre STRICT des étapes, dans chaque micro-batch :
 *   1. FETCH      : readStream binaryFile sur DEST/continuous_ingress/
 *   2. PARSE +
 *      TRANSFORM  : nettoyage (filtre images valides) + typage des colonnes
 *   3. WRITE      : écriture immédiate des images propres dans STAGING/
 *                   (Parquet — couche d'historique)
 *   4. PREDICT    : modèle de vision ONNX broadcasté (injecté au démarrage),
 *                   scoring DIRECTEMENT sur les octets de l'image
 *   5. WRITE      : résultat final + confidence_score dans OUTPUT/
 */
object ConsumerApp {

  /**
   * PARSE & TRANSFORM : nettoie et type le flux brut binaryFile.
   * Produit exactement Schemas.stagingSchema — une ligne par image propre.
   */
  private def parseAndTransform(raw: DataFrame): DataFrame =
    raw
      .where(col("path").rlike("(?i)\\.(jpg|jpeg|png)$")) // images uniquement
      .where(col("content").isNotNull && col("length") > 0) // écarte les fichiers vides
      .select(
        regexp_extract(col("path"), "(?i)([^/\\\\]+)\\.(jpg|jpeg|png)$", 1).as("image_name"),
        col("path").as("image_path"),
        col("content").as("image_data"),
        col("length"),
        col("modificationTime").as("ingested_at"),
        current_timestamp().as("processed_at"))

  def main(args: Array[String]): Unit = {
    val ingressPath    = if (args.length > 0) args(0) else "data/dest/continuous_ingress"
    val stagingPath    = if (args.length > 1) args(1) else "data/staging"
    val outputPath     = if (args.length > 2) args(2) else "data/output"
    val modelPath      = if (args.length > 3) args(3) else "data/models"
    val checkpointPath = if (args.length > 4) args(4) else "data/checkpoint"

    val spark = SparkSession.builder()
      .appName("PlateDetection-Consumer-Streaming")
      .getOrCreate()

    // Injection du modèle : broadcast une seule fois au démarrage
    ModelLoader.initialize(spark, modelPath.stripSuffix("/").stripSuffix("\\") + "/")

    // 1. FETCH — stream permanent sur les images brutes déposées par le Producer
    val rawStream = spark.readStream
      .format("binaryFile")
      .schema(Schemas.binaryFileSchema) // schéma fixe imposé par la source binaryFile
      .option("recursiveFileLookup", "true")
      .option("maxFilesPerTrigger", 100)
      .load(ingressPath)

    val query = rawStream.writeStream
      .foreachBatch { (batchDF: DataFrame, batchId: Long) =>
        // 2. PARSE & TRANSFORM — persist : évite de recalculer le batch
        //    pour les deux écritures + le scoring
        val staged = parseAndTransform(batchDF).persist()
        val rows   = staged.count()

        if (rows > 0) {
          // 3. WRITE intermédiaire — couche staging (historique des images propres)
          staged.write
            .mode("append")
            .option("compression", "snappy")
            .parquet(stagingPath)

          // 4. PREDICT — vision ONNX directement sur les octets de l'image
          val scored = ModelLoader.score(staged)

          // 5. WRITE OUTPUT — prédiction + confidence, lus par le dashboard
          scored.write
            .mode("append")
            .option("compression", "snappy")
            .parquet(outputPath)

          println(s"[Consumer] batch $batchId : $rows images → staging + output")
        }
        staged.unpersist()
        () // retour Unit explicite : lève l'ambiguïté d'overload de foreachBatch (Scala 2.13)
      }
      .option("checkpointLocation",
        checkpointPath.stripSuffix("/").stripSuffix("\\") + "/consumer")
      .trigger(Trigger.ProcessingTime("10 seconds"))
      .start()

    query.awaitTermination()
  }
}
