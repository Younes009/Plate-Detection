package com.project.producer

import com.project.common._
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

import scala.util.Try

/**
 *
 * Microservice 1 — Producer BATCH (Spark Core + Scala).
 *
 * Args : [sourcePath] [destPath] [batchSize] [intervalSeconds]
 *
 * Logique :
 *   1. Parcourt récursivement sourcePath pour lister toutes les images (JPEG/PNG)
 *   2. Découpe en batches de batchSize
 *   3. Pour chaque batch :
 *        - Lit images via binaryFile
 *        - Dérive le chemin du label YOLO (/images/ → /labels/, .jpg → .txt)
 *        - Lit le contenu du label si présent (sinon null)
 *        - Écrit Parquet Snappy dans destPath/batch_N_TS/
 *   4. Attend intervalSeconds entre chaque batch
 *   5. Arrêt automatique après le dernier batch
 */
object ProducerApp {

  private def listImages(fs: FileSystem, path: Path): Seq[String] =
    try {
      fs.listStatus(path).flatMap { s =>
        if (s.isDirectory) listImages(fs, s.getPath)
        else if (s.getPath.getName.toLowerCase.matches(".*\\.(jpg|jpeg|png)$"))
          Seq(s.getPath.toString)
        else Seq.empty
      }
    } catch { case _: Exception => Seq.empty }

  private def ensureDir(spark: SparkSession, pathStr: String): FileSystem = {
    val p  = new Path(pathStr)
    val fs = p.getFileSystem(spark.sparkContext.hadoopConfiguration)
    if (!fs.exists(p)) { fs.mkdirs(p); println(s"[Producer] Créé : $pathStr") }
    fs
  }

  def main(args: Array[String]): Unit = {
    val sourcePath      = if (args.length > 0) args(0) else "license-plate-detection-dataset-10125-images"
    val destPath        = if (args.length > 1) args(1) else "data/dest"
    val batchSize       = if (args.length > 2) args(2).toInt  else 100
    val intervalSeconds = if (args.length > 3) args(3).toLong else 0L

    val spark = SparkSession.builder()
      .appName("PlateDetection-Producer-Batch")
      .getOrCreate()

    val fs = ensureDir(spark, sourcePath)
    ensureDir(spark, destPath)

    val allImages = listImages(fs, new Path(sourcePath))
    println(s"[Producer] ${allImages.size} images trouvées dans $sourcePath")

    if (allImages.isEmpty) {
      println("[Producer] Aucune image — arrêt.")
      spark.stop()
      return
    }

    // image URI (file:///path/images/foo.jpg) → label URI (file:///path/labels/foo.txt)
    val toLabelUri = udf { imgUri: String =>
      if (imgUri == null) null
      else imgUri
        .replaceAll("(?i)/images/", "/labels/")
        .replaceAll("(?i)\\.(jpg|jpeg|png)$", ".txt")
    }

    // Lit le fichier texte pointé par un URI file:/// — null si absent
    val readLabel = udf { labelUri: String =>
      if (labelUri == null) null
      else Try {
        val file = new java.io.File(new java.net.URI(labelUri))
        if (file.exists()) new String(java.nio.file.Files.readAllBytes(file.toPath), "UTF-8")
        else null
      }.getOrElse(null): String
    }

    val batches = allImages.grouped(batchSize).toSeq

    batches.zipWithIndex.foreach { case (batchPaths, idx) =>
      val ts   = System.currentTimeMillis()
      val dest = s"$destPath/batch_${idx}_$ts"

      println(s"[Producer] Batch $idx/${batches.size - 1} — ${batchPaths.size} images")

      spark.read.format("binaryFile")
        .load(batchPaths: _*)
        .withColumn("image_name",
          regexp_extract(col("path"), "(?i)([^/\\\\]+)\\.(jpg|jpeg|png)$", 1))
        .withColumn("split",
          when(col("path").rlike("/(train|valid|test)/"),
               regexp_extract(col("path"), "/(train|valid|test)/", 1))
          .otherwise(lit("unknown")))
        .withColumn("ingested_at", current_timestamp())
        .withColumn("label_path", toLabelUri(col("path")))
        .withColumn("label_data",  readLabel(col("label_path")))
        .select(
          col("image_name"),
          col("split"),
          col("path").as("image_path"),
          col("content").as("image_data"),
          col("label_path"),
          col("label_data"),
          col("ingested_at")
        )
        .write
        .mode("overwrite")
        .option("compression", "snappy")
        .parquet(dest)

      println(s"[Producer] Batch $idx écrit → $dest")

      if (idx < batches.size - 1 && intervalSeconds > 0)
        Thread.sleep(intervalSeconds * 1000L)
    }

    println(s"[Producer] Terminé — ${batches.size} batches dans $destPath")
    spark.stop()
  }
}
