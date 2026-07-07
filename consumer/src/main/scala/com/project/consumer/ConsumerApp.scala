package com.project.consumer

import com.project.common.Schemas
import com.project.ml.ModelLoader
import org.apache.hadoop.fs.Path
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
  private[consumer] def parseAndTransform(raw: DataFrame): DataFrame =
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

  /**
   * Écriture IDEMPOTENTE d'un micro-batch, PARTITIONNÉE PAR DATE :
   *   <base>/ingest_date=YYYY-MM-DD/batch_id=N/
   *
   * - Idempotence : le checkpoint garantit que la source ne relit pas un
   *   fichier, mais pas l'idempotence du sink — sur reprise après échec, un
   *   même batchId est rejoué et .mode("append") dupliquerait les lignes.
   *   Le rejeu est détecté par globStatus sur TOUTES les dates (un crash à
   *   cheval sur minuit ne réécrit pas le batch sous la nouvelle date).
   *   _SUCCESS présent → skip ; répertoire orphelin → purgé puis réécrit.
   * - Partition par date : évite l'accumulation à plat de milliers de
   *   répertoires batch_id (dégradation du listing) et permet la rétention /
   *   compaction par jour. Spark relit ingest_date et batch_id comme colonnes
   *   de partition.
   */
  private[consumer] def writeIdempotent(df: DataFrame, basePath: String, batchId: Long): Boolean = {
    val base = new Path(basePath.stripSuffix("/").stripSuffix("\\"))
    val fs   = base.getFileSystem(df.sparkSession.sparkContext.hadoopConfiguration)

    val replayed = fs.globStatus(new Path(base, s"ingest_date=*/batch_id=$batchId/_SUCCESS"))
    if (replayed != null && replayed.nonEmpty) {
      println(s"[Consumer] batch $batchId déjà présent dans $basePath — skip (rejeu).")
      false
    } else {
      val dir = new Path(base, s"ingest_date=${java.time.LocalDate.now}/batch_id=$batchId")
      if (fs.exists(dir)) fs.delete(dir, true) // batch interrompu : on repart proprement
      df.write
        .mode("overwrite")
        .option("compression", "snappy")
        .parquet(dir.toString)
      true
    }
  }

  /**
   * Archivage de DEST après persistance : une fois le batch écrit dans
   * staging + output, les fichiers sources sont DÉPLACÉS de
   * DEST/continuous_ingress vers ARCHIVE (rename, pas de copie). Sans ça,
   * DEST grossit indéfiniment et le listing de la source streaming ralentit
   * à chaque trigger. Les octets restent disponibles dans STAGING
   * (image_data) et dans l'archive (aperçu dashboard).
   */
  private[consumer] def archiveIngested(df: DataFrame, archivePath: String): Unit = {
    val conf    = df.sparkSession.sparkContext.hadoopConfiguration
    val archive = new Path(archivePath.stripSuffix("/").stripSuffix("\\"))
    val paths   = df.select("image_path").collect().map(_.getString(0))
    if (paths.isEmpty) return
    val fs = archive.getFileSystem(conf)
    if (!fs.exists(archive)) fs.mkdirs(archive)
    var moved = 0
    paths.foreach { uri =>
      val src = new Path(uri)
      if (fs.exists(src) && fs.rename(src, new Path(archive, src.getName))) moved += 1
    }
    if (moved > 0) println(s"[Consumer] $moved fichiers archivés → $archive")
  }

  // Version Spark contre laquelle les jars sont compilés (build.sbt) : un
  // runtime différent produit des NoSuchMethodError sur l'API streaming.
  private val ExpectedSparkPrefix = "4.1"

  def main(args: Array[String]): Unit = {
    val ingressPath    = if (args.length > 0) args(0) else "data/dest/continuous_ingress"
    val stagingPath    = if (args.length > 1) args(1) else "data/staging"
    val outputPath     = if (args.length > 2) args(2) else "data/output"
    val modelPath      = if (args.length > 3) args(3) else "data/models"
    val checkpointPath = if (args.length > 4) args(4) else "data/checkpoint"
    val archivePath    = if (args.length > 5) args(5) else "data/archive/continuous_ingress"

    val spark = SparkSession.builder()
      .appName("PlateDetection-Consumer-Streaming")
      .getOrCreate()

    if (!spark.version.startsWith(ExpectedSparkPrefix))
      System.err.println(
        s"[Consumer][WARN] Spark runtime ${spark.version} ≠ build $ExpectedSparkPrefix.x " +
        "(build.sbt) — risque de NoSuchMethodError. Aligner spark-submit et sparkVersion.")

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
          writeIdempotent(staged, stagingPath, batchId)

          // 4. PREDICT — vision ONNX directement sur les octets de l'image
          val scored = ModelLoader.score(staged)

          // 5. WRITE OUTPUT — prédiction + confidence, lus par le dashboard
          if (writeIdempotent(scored, outputPath, batchId))
            println(s"[Consumer] batch $batchId : $rows images → staging + output")

          // 6. ARCHIVE — DEST purgé seulement APRÈS persistance staging+output
          archiveIngested(staged, archivePath)
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
