package com.project.producer

import org.apache.hadoop.fs.{FileSystem, FileUtil, Path}
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Microservice 1 — Producer (Spark Core + Hadoop FileSystem API only).
 *
 * Short-lived job in both modes: transfer, then exit(0). An external trigger
 * (main.py scheduler, cron) relaunches it — it never loops.
 *
 *  - "continuous": scan SOURCE/continuous_ingress (raw images only, no YOLO
 *    labels there) and MOVE everything to DEST/continuous_ingress in lots of
 *    batchSize. Moving empties the source: each image is ingested once.
 *  - "batch" (weekly): full recursive COPY of SOURCE/ml_dataset to
 *    DEST/ml_dataset, train/test/val tree preserved (images AND labels).
 *    Idempotent: already-copied files are skipped.
 *
 * Raw byte-for-byte transfer — no preprocessing, no conversion.
 *
 * Args: [sourcePath] [destPath] [batchSize] [intervalSeconds] [recursive] [mode]
 */
object ProducerApp {

  private val ImagePattern = "(?i).*\\.(jpg|jpeg|png)$"

  /** List files under dir via Hadoop FS, recursively if asked. */
  private def listFiles(fs: FileSystem, dir: Path, recursive: Boolean): Seq[Path] = {
    if (!fs.exists(dir)) return Seq.empty
    val (dirs, files) = fs.listStatus(dir).toSeq.partition(_.isDirectory)
    val here = files.map(_.getPath)
    if (recursive) here ++ dirs.flatMap(d => listFiles(fs, d.getPath, recursive))
    else here
  }

  def main(args: Array[String]): Unit = {
    val sourcePath      = if (args.length > 0) args(0) else "data/source/ml_dataset"
    val destPath        = if (args.length > 1) args(1) else "data/dest/ml_dataset"
    val batchSize       = if (args.length > 2) args(2).toInt     else 100
    val intervalSeconds = if (args.length > 3) args(3).toLong    else 0L
    val recursive       = if (args.length > 4) args(4).toBoolean else true
    val mode            = if (args.length > 5) args(5)           else "batch"
    require(Set("batch", "continuous").contains(mode),
      s"[Producer] unknown mode: $mode (expected: batch | continuous)")

    val sc = new SparkContext(
      new SparkConf()
        .setAppName(s"PlateDetection-Producer-$mode")
        .setIfMissing("spark.master", "local[2]"))
    val hadoopConf = sc.hadoopConfiguration

    val srcRoot = new Path(sourcePath)
    val srcFs   = srcRoot.getFileSystem(hadoopConf)
    val srcQ    = srcFs.makeQualified(srcRoot)
    require(srcFs.exists(srcQ), s"[Producer] source not found: $srcQ")

    val destRoot = new Path(destPath)
    val destFs   = destRoot.getFileSystem(hadoopConf)
    val destQ    = destFs.makeQualified(destRoot)
    if (!destFs.exists(destQ)) destFs.mkdirs(destQ)

    val srcPrefix = srcQ.toString
    def relativePath(p: Path): String =
      p.toString.stripPrefix(srcPrefix).stripPrefix("/")

    // move=true deletes the source (continuous); move=false skips existing (weekly, replayable)
    def transferRaw(src: Path, move: Boolean): Boolean = {
      val dst = new Path(destQ, relativePath(src))
      if (!move && destFs.exists(dst)) false
      else {
        FileUtil.copy(srcFs, src, destFs, dst, /*deleteSource=*/ move, /*overwrite=*/ move, hadoopConf)
        true
      }
    }

    // continuous: images only (no labels in the ingress); batch: everything
    val move  = mode == "continuous"
    val files = {
      val all = listFiles(srcFs, srcQ, recursive)
      if (move) all.filter(_.getName.matches(ImagePattern)) else all
    }
    println(s"[Producer:$mode] ${files.size} files to transfer from $srcQ")

    if (files.isEmpty) {
      println(s"[Producer:$mode] nothing to transfer — exit.")
      sc.stop()
      return
    }

    // transfer in lots, then STOP — the job is ephemeral in both modes
    val batches = files.grouped(batchSize).toSeq
    batches.zipWithIndex.foreach { case (batch, idx) =>
      val n = batch.count(transferRaw(_, move))
      println(s"[Producer:$mode] lot ${idx + 1}/${batches.size}: $n files -> $destQ")
      if (idx < batches.size - 1 && intervalSeconds > 0)
        Thread.sleep(intervalSeconds * 1000L)
    }

    println(s"[Producer:$mode] done (${files.size} files) — exit.")
    sc.stop()
    sys.exit(0) // ephemeral job: no thread survives the transfer
  }
}
