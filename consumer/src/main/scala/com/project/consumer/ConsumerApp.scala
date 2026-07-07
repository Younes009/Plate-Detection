package com.project.consumer

import com.project.common.Schemas
import com.project.ml.ModelLoader
import org.apache.hadoop.fs.Path
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.streaming.Trigger

/**
 * Microservice 2 — Consumer & Predictor (Spark Structured Streaming).
 *
 * The only permanent service (awaitTermination, exactly-once checkpoint).
 * Native streaming: no DStream, no manual loop. Listens EXCLUSIVELY to
 * DEST/continuous_ingress — raw images without YOLO labels; ml_dataset never
 * enters the stream.
 *
 * Strict order per micro-batch:
 *   1. FETCH   readStream binaryFile on DEST/continuous_ingress
 *   2. PARSE + TRANSFORM   filter valid images, typed columns
 *   3. WRITE   staging (Parquet history) BEFORE any prediction
 *   4. PREDICT broadcast ONNX vision model, scoring on raw image bytes
 *   5. WRITE   prediction + confidence + predicted bbox to OUTPUT
 *   6. ARCHIVE move processed files out of DEST
 *
 * Args: [ingressPath] [stagingPath] [outputPath] [modelPath] [checkpointPath] [archivePath]
 */
object ConsumerApp {

  /** PARSE & TRANSFORM: clean and type the raw binaryFile stream.
   *  Produces exactly Schemas.stagingSchema, one row per valid image. */
  private[consumer] def parseAndTransform(raw: DataFrame): DataFrame =
    raw
      .where(col("path").rlike("(?i)\\.(jpg|jpeg|png)$"))
      .where(col("content").isNotNull && col("length") > 0)
      .select(
        regexp_extract(col("path"), "(?i)([^/\\\\]+)\\.(jpg|jpeg|png)$", 1).as("image_name"),
        col("path").as("image_path"),
        col("content").as("image_data"),
        col("length"),
        col("modificationTime").as("ingested_at"),
        current_timestamp().as("processed_at"))

  /**
   * Idempotent micro-batch write, partitioned by date:
   *   <base>/ingest_date=YYYY-MM-DD/batch_id=N/
   *
   * The checkpoint stops the source from re-reading files, not the sink from
   * replaying a batchId after a crash — append mode would duplicate rows.
   * Replay detection globs batch_id=N across ALL dates (midnight-crossing
   * crash safe). _SUCCESS present -> skip; orphan dir -> purge and rewrite.
   * Date partitioning also keeps directory listings bounded.
   */
  private[consumer] def writeIdempotent(df: DataFrame, basePath: String, batchId: Long): Boolean = {
    val base = new Path(basePath.stripSuffix("/").stripSuffix("\\"))
    val fs   = base.getFileSystem(df.sparkSession.sparkContext.hadoopConfiguration)

    val replayed = fs.globStatus(new Path(base, s"ingest_date=*/batch_id=$batchId/_SUCCESS"))
    if (replayed != null && replayed.nonEmpty) {
      println(s"[Consumer] batch $batchId already in $basePath — skip (replay).")
      false
    } else {
      val dir = new Path(base, s"ingest_date=${java.time.LocalDate.now}/batch_id=$batchId")
      if (fs.exists(dir)) fs.delete(dir, true) // interrupted batch leftovers
      df.write
        .mode("overwrite")
        .option("compression", "snappy")
        .parquet(dir.toString)
      true
    }
  }

  /** Move processed source files from DEST to ARCHIVE (rename, no copy).
   *  Called only AFTER staging+output are persisted. Keeps the streaming
   *  source listing fast; bytes remain in staging and in the archive. */
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
    if (moved > 0) println(s"[Consumer] $moved files archived -> $archive")
  }

  // Spark version the jars were built against; a different runtime throws
  // NoSuchMethodError on the streaming API.
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
        s"[Consumer][WARN] Spark runtime ${spark.version} != build $ExpectedSparkPrefix.x — " +
        "NoSuchMethodError risk. Align spark-submit with build.sbt.")

    // model injected once at startup, broadcast to executors
    ModelLoader.initialize(spark, modelPath.stripSuffix("/").stripSuffix("\\") + "/")

    // 1. FETCH — permanent stream on raw images dropped by the Producer
    val rawStream = spark.readStream
      .format("binaryFile")
      .schema(Schemas.binaryFileSchema)
      .option("recursiveFileLookup", "true")
      .option("maxFilesPerTrigger", 100)
      .load(ingressPath)

    val query = rawStream.writeStream
      .foreachBatch { (batchDF: DataFrame, batchId: Long) =>
        // 2. PARSE & TRANSFORM — persist: reused by two writes + scoring
        val staged = parseAndTransform(batchDF).persist()
        val rows   = staged.count()

        if (rows > 0) {
          // 3. WRITE staging (history of clean images)
          writeIdempotent(staged, stagingPath, batchId)

          // 4. PREDICT — ONNX directly on image bytes
          val scored = ModelLoader.score(staged)

          // 5. WRITE output — read by the dashboard
          if (writeIdempotent(scored, outputPath, batchId))
            println(s"[Consumer] batch $batchId: $rows images -> staging + output")

          // 6. ARCHIVE — only after staging+output are persisted
          archiveIngested(staged, archivePath)
        }
        staged.unpersist()
        () // explicit Unit: resolves foreachBatch overload ambiguity (Scala 2.13)
      }
      .option("checkpointLocation",
        checkpointPath.stripSuffix("/").stripSuffix("\\") + "/consumer")
      .trigger(Trigger.ProcessingTime("10 seconds"))
      .start()

    query.awaitTermination()
  }
}
