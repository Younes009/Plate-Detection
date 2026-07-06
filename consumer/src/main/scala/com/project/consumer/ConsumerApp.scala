package com.project.consumer

import com.project.common.Schemas
import com.project.ml.ModelLoader
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.streaming.Trigger

import scala.util.Try

/**
 * Microservice 2 — Consumer & Predictor (Spark Structured Streaming + Scala).
 *
 * Args : [destPath] [stagingPath] [outputPath] [modelPath] [checkpointPath]
 *
 * Ordre STRICT des étapes, dans chaque micro-batch :
 *   1. FETCH      : readStream binaryFile sur DEST/ (fichiers bruts du Producer)
 *   2. PARSE +
 *      TRANSFORM  : nom / split / label YOLO → colonnes typées + features bbox
 *   3. WRITE (4)  : écriture immédiate des données propres dans STAGING/
 *                   (couche lue par le service ML Python — READ 5)
 *   4. PREDICT    : modèle ONNX/MLlib injecté via ModelLoader → prediction + score
 *   5. WRITE      : résultats enrichis dans OUTPUT/ (lus par Streamlit — READ 8)
 *
 * awaitTermination() : écoute permanente, exactly-once garanti par le checkpoint.
 * Pas de boucle manuelle, pas de DStream — Structured Streaming natif uniquement.
 */
object ConsumerApp {

  // image URI (…/images/foo.jpg) → label URI (…/labels/foo.txt)
  private val toLabelUri = udf { imgUri: String =>
    if (imgUri == null) null
    else imgUri
      .replaceAll("(?i)/images/", "/labels/")
      .replaceAll("(?i)\\.(jpg|jpeg|png)$", ".txt")
  }

  // Lit le contenu du label via l'API Hadoop FS (fonctionne en local ET sur HDFS) — null si absent
  private val readLabel = udf { labelUri: String =>
    if (labelUri == null) null
    else Try {
      val p  = new Path(labelUri)
      val fs = p.getFileSystem(new Configuration())
      if (!fs.exists(p)) null
      else {
        val in = fs.open(p)
        try new String(org.apache.commons.io.IOUtils.toByteArray(in), "UTF-8")
        finally in.close()
      }
    }.getOrElse(null): String
  }

  /**
   * PARSE & TRANSFORM : nettoie, type et enrichit le flux brut binaryFile.
   * Une ligne par bounding box YOLO (explode_outer garde les images sans label).
   * Produit exactement Schemas.stagingSchema.
   */
  private def parseAndTransform(raw: DataFrame): DataFrame = {
    val yoloField = (i: Int) =>
      when(size(col("yolo_tokens")) >= 5, element_at(col("yolo_tokens"), i))

    raw
      .where(col("path").rlike("(?i)\\.(jpg|jpeg|png)$")) // ignore les .txt copiés dans DEST
      .withColumn("image_name",
        regexp_extract(col("path"), "(?i)([^/\\\\]+)\\.(jpg|jpeg|png)$", 1))
      .withColumn("split",
        when(col("path").rlike("[/\\\\](train|valid|test)[/\\\\]"),
             regexp_extract(col("path"), "[/\\\\](train|valid|test)[/\\\\]", 1))
        .otherwise(lit("unknown")))
      .withColumn("label_path", toLabelUri(col("path")))
      .withColumn("label_data", readLabel(col("label_path")))
      .withColumn("yolo_line",   explode_outer(split(trim(col("label_data")), "\n")))
      .withColumn("yolo_tokens", split(trim(col("yolo_line")), "\\s+"))
      .withColumn("class_id", yoloField(1).cast("int"))
      .withColumn("cx_norm",  yoloField(2).cast("double"))
      .withColumn("cy_norm",  yoloField(3).cast("double"))
      .withColumn("w_norm",   yoloField(4).cast("double"))
      .withColumn("h_norm",   yoloField(5).cast("double"))
      .withColumn("bbox_area_norm", col("w_norm") * col("h_norm"))
      .withColumn("aspect_ratio",
        when(col("h_norm") > 0, col("w_norm") / col("h_norm")))
      .select(
        col("image_name"),
        col("split"),
        col("path").as("image_path"),
        col("content").as("image_data"),
        col("modificationTime").as("ingested_at"),
        col("label_path"),
        col("label_data"),
        col("class_id"),
        col("cx_norm"), col("cy_norm"), col("w_norm"), col("h_norm"),
        col("bbox_area_norm"), col("aspect_ratio"),
        current_timestamp().as("processed_at"))
  }

  def main(args: Array[String]): Unit = {
    val destPath       = if (args.length > 0) args(0) else "data/dest"
    val stagingPath    = if (args.length > 1) args(1) else "data/staging"
    val outputPath     = if (args.length > 2) args(2) else "data/output"
    val modelPath      = if (args.length > 3) args(3) else "data/models"
    val checkpointPath = if (args.length > 4) args(4) else "data/checkpoint"

    val spark = SparkSession.builder()
      .appName("PlateDetection-Consumer-Streaming")
      .getOrCreate()

    // Injection du modèle : broadcast une seule fois au démarrage
    ModelLoader.initialize(spark, modelPath.stripSuffix("/").stripSuffix("\\") + "/")

    // 1. FETCH — stream continu sur les fichiers bruts déposés par le Producer
    val rawStream = spark.readStream
      .format("binaryFile")
      .schema(Schemas.binaryFileSchema) // schéma fixe imposé par la source binaryFile
      .option("recursiveFileLookup", "true")
      .option("maxFilesPerTrigger", 100)
      .load(destPath)

    val query = rawStream.writeStream
      .foreachBatch { (batchDF: DataFrame, batchId: Long) =>
        // 2. PARSE & TRANSFORM — persist : évite de recalculer le batch
        //    pour chacune des deux écritures + le scoring
        val staged = parseAndTransform(batchDF).persist()
        val rows   = staged.count()

        if (rows > 0) {
          // 3. WRITE (4) — couche staging, lue par le service ML (READ 5)
          staged.write
            .mode("append")
            .option("compression", "snappy")
            .parquet(stagingPath)

          // 4. PREDICT — scoring sur les données fraîchement préparées
          val scored = ModelLoader.score(staged)

          // 5. WRITE OUTPUT — résultats enrichis, lus par Streamlit (READ 8)
          scored.write
            .mode("append")
            .option("compression", "snappy")
            .parquet(outputPath)

          println(s"[Consumer] batch $batchId : $rows lignes → staging + output")
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
