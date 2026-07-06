package com.project.producer

import org.apache.hadoop.fs.{FileSystem, FileUtil, Path}
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Microservice 1 — Producer BATCH (Spark Core + Scala).
 *
 * Args : [sourcePath] [destPath] [batchSize] [intervalSeconds] [recursive]
 *
 * Transfert BRUT : copie les fichiers images (et leur label YOLO .txt associé)
 * depuis SOURCE vers DEST via l'API Hadoop FileSystem exclusivement.
 * AUCUN prétraitement, AUCUNE conversion de format, AUCUNE colonne dérivée —
 * les fichiers arrivent dans DEST identiques à l'octet près, en conservant
 * l'arborescence relative (train/images/x.jpg → DEST/train/images/x.jpg).
 *
 * Découpage en lots de batchSize images ; pause de intervalSeconds entre deux
 * lots (aucune consommation de ressources pendant l'attente) ; le service
 * s'arrête de lui-même une fois tous les lots envoyés.
 */
object ProducerApp {

  private val ImagePattern = "(?i).*\\.(jpg|jpeg|png)$"

  /** Liste les fichiers sous dir via l'API Hadoop FS, récursivement si demandé. */
  private def listFiles(fs: FileSystem, dir: Path, recursive: Boolean): Seq[Path] = {
    val (dirs, files) = fs.listStatus(dir).toSeq.partition(_.isDirectory)
    val here = files.map(_.getPath)
    if (recursive) here ++ dirs.flatMap(d => listFiles(fs, d.getPath, recursive))
    else here
  }

  def main(args: Array[String]): Unit = {
    val sourcePath      = if (args.length > 0) args(0) else "license-plate-detection-dataset-10125-images"
    val destPath        = if (args.length > 1) args(1) else "data/dest"
    val batchSize       = if (args.length > 2) args(2).toInt     else 100
    val intervalSeconds = if (args.length > 3) args(3).toLong    else 0L
    val recursive       = if (args.length > 4) args(4).toBoolean else true

    val sc = new SparkContext(
      new SparkConf()
        .setAppName("PlateDetection-Producer-Batch")
        .setIfMissing("spark.master", "local[2]"))
    val hadoopConf = sc.hadoopConfiguration

    val srcRoot = new Path(sourcePath)
    val srcFs   = srcRoot.getFileSystem(hadoopConf)
    val srcQ    = srcFs.makeQualified(srcRoot)
    require(srcFs.exists(srcQ), s"[Producer] Dossier source introuvable : $srcQ")

    val destRoot = new Path(destPath)
    val destFs   = destRoot.getFileSystem(hadoopConf)
    val destQ    = destFs.makeQualified(destRoot)
    if (!destFs.exists(destQ)) destFs.mkdirs(destQ)

    val images = listFiles(srcFs, srcQ, recursive)
      .filter(_.getName.matches(ImagePattern))
    println(s"[Producer] ${images.size} images trouvées dans $srcQ (recursive=$recursive)")

    if (images.isEmpty) {
      println("[Producer] Aucune image — arrêt.")
      sc.stop()
      return
    }

    val srcPrefix = srcQ.toString
    def relativePath(p: Path): String =
      p.toString.stripPrefix(srcPrefix).stripPrefix("/")

    // Label YOLO associé : .../images/foo.jpg → .../labels/foo.txt (s'il existe)
    def labelFor(img: Path): Option[Path] = {
      val candidate = new Path(
        img.toString
          .replace("/images/", "/labels/")
          .replaceAll("(?i)\\.(jpg|jpeg|png)$", ".txt"))
      if (srcFs.exists(candidate)) Some(candidate) else None
    }

    /** Copie brute octet-à-octet via Hadoop FS ; ignore si déjà présent (rejouable). */
    def copyRaw(src: Path): Boolean = {
      val dst = new Path(destQ, relativePath(src))
      if (destFs.exists(dst)) false
      else {
        FileUtil.copy(srcFs, src, destFs, dst, /*deleteSource=*/ false, /*overwrite=*/ false, hadoopConf)
        true
      }
    }

    val batches = images.grouped(batchSize).toSeq
    batches.zipWithIndex.foreach { case (batch, idx) =>
      println(s"[Producer] Batch ${idx + 1}/${batches.size} — ${batch.size} images")

      // Label copié AVANT son image : quand le Consumer voit l'image, le label est déjà là.
      val copied = batch.count { img =>
        labelFor(img).foreach(copyRaw)
        copyRaw(img)
      }
      println(s"[Producer] Batch ${idx + 1} : $copied nouvelles images → $destQ")

      if (idx < batches.size - 1 && intervalSeconds > 0)
        Thread.sleep(intervalSeconds * 1000L)
    }

    println(s"[Producer] Terminé — ${batches.size} batches envoyés dans $destQ")
    sc.stop()
  }
}
