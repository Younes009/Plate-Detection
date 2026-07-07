package com.project.producer

import org.apache.hadoop.fs.{FileSystem, FileUtil, Path}
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Microservice 1 — Producer (Spark Core + API Hadoop FileSystem EXCLUSIVEMENT).
 *
 * JOB ÉPHÉMÈRE dans les deux modes : on le lance, il transfère, il S'ARRÊTE
 * TOUT SEUL. Aucune boucle de veille, aucun processus permanent — c'est un
 * déclencheur externe (orchestrateur, cron, trigger) qui le relance.
 *
 *   - mode "continuous" (flux d'inférence, relancé régulièrement) :
 *       scanne SOURCE/continuous_ingress/ (images brutes UNIQUEMENT — aucun
 *       label YOLO ici) et DÉPLACE tout ce qui s'y trouve vers
 *       DEST/continuous_ingress/, par lots de batchSize espacés de
 *       intervalSeconds. Le déplacement vide la source : chaque image n'est
 *       ingérée qu'une fois. Puis exit.
 *
 *   - mode "batch" (flux hebdomadaire, lancé une fois par semaine) :
 *       copie BRUTE, RÉCURSIVE et INTÉGRALE de SOURCE/ml_dataset/ vers
 *       DEST/ml_dataset/ en préservant l'arborescence train/test/val (images
 *       ET labels YOLO). Idempotent : les fichiers déjà copiés sont ignorés.
 *       Puis exit.
 *
 * AUCUN prétraitement, AUCUNE conversion : transfert octet-à-octet.
 *
 * Args : [sourcePath] [destPath] [batchSize] [intervalSeconds] [recursive] [mode]
 *   mode ∈ {batch, continuous} — défaut : batch
 */
object ProducerApp {

  private val ImagePattern = "(?i).*\\.(jpg|jpeg|png)$"

  /** Liste les fichiers sous dir via l'API Hadoop FS, récursivement si demandé. */
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
      s"[Producer] mode inconnu : $mode (attendu : batch | continuous)")

    val sc = new SparkContext(
      new SparkConf()
        .setAppName(s"PlateDetection-Producer-$mode")
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

    val srcPrefix = srcQ.toString
    def relativePath(p: Path): String =
      p.toString.stripPrefix(srcPrefix).stripPrefix("/")

    /**
     * Transfert brut octet-à-octet via Hadoop FS, arborescence relative préservée.
     * move=true : déplace (supprime la source) — flux continu.
     * move=false : copie, ignore si déjà présent — flux hebdo rejouable.
     */
    def transferRaw(src: Path, move: Boolean): Boolean = {
      val dst = new Path(destQ, relativePath(src))
      if (!move && destFs.exists(dst)) false
      else {
        FileUtil.copy(srcFs, src, destFs, dst, /*deleteSource=*/ move, /*overwrite=*/ move, hadoopConf)
        true
      }
    }

    // Sélection des fichiers selon le mode :
    //   continuous → images brutes seulement (l'ingress ne contient pas de labels)
    //   batch      → TOUT (images + labels YOLO, arborescence intégrale)
    val move  = mode == "continuous"
    val files = {
      val all = listFiles(srcFs, srcQ, recursive)
      if (move) all.filter(_.getName.matches(ImagePattern)) else all
    }
    println(s"[Producer:$mode] ${files.size} fichiers à transférer depuis $srcQ")

    if (files.isEmpty) {
      println(s"[Producer:$mode] Rien à transférer — arrêt.")
      sc.stop()
      return
    }

    // Transfert par lots puis ARRÊT — le job est éphémère dans les deux modes.
    val batches = files.grouped(batchSize).toSeq
    batches.zipWithIndex.foreach { case (batch, idx) =>
      val n = batch.count(transferRaw(_, move))
      println(s"[Producer:$mode] Lot ${idx + 1}/${batches.size} : $n fichiers → $destQ")
      if (idx < batches.size - 1 && intervalSeconds > 0)
        Thread.sleep(intervalSeconds * 1000L)
    }

    println(s"[Producer:$mode] Terminé (${files.size} fichiers) — arrêt du job.")
    sc.stop()
    // Job éphémère : extinction propre et explicite (code retour 0),
    // aucun thread ne doit survivre au transfert.
    sys.exit(0)
  }
}
