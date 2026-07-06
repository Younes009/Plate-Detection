package com.project.producer

import org.apache.hadoop.fs.{FileSystem, FileUtil, Path}
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Microservice 1 — Producer (Spark Core + API Hadoop FileSystem EXCLUSIVEMENT).
 *
 * Gère les deux temporalités du projet, de manière étanche :
 *
 *   - mode "batch" (flux HEBDOMADAIRE — réentraînement) :
 *       copie BRUTE et RÉCURSIVE de l'INTÉGRALITÉ de SOURCE/ml_dataset/
 *       (arborescence train/test/val préservée à l'identique, labels compris)
 *       vers DEST/ml_dataset/. Lot par lot, puis s'arrête de lui-même.
 *       Le service ML Python lit ensuite cette copie pour se réentraîner.
 *
 *   - mode "continuous" (flux CONTINU — inférence au fil de l'eau) :
 *       surveille SOURCE/continuous_ingress/ en permanence et DÉPLACE les
 *       nouvelles images (avec leur label YOLO s'il existe) vers
 *       DEST/continuous_ingress/, par lots de batchSize toutes les
 *       intervalSeconds. Le déplacement vide la source : chaque fichier n'est
 *       ingéré qu'une fois. C'est le Consumer (Structured Streaming) qui
 *       détecte ensuite les fichiers déposés dans DEST.
 *
 * AUCUN prétraitement, AUCUNE conversion : transfert octet-à-octet.
 * NB : la boucle de veille du mode continu est le rôle même de ce service
 * (déplacement fichier) ; l'interdiction des boucles manuelles concerne le
 * moteur de streaming du Microservice 2, qui reste du Structured Streaming natif.
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
    val intervalSeconds = if (args.length > 3) args(3).toLong    else 5L
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

    // Label YOLO associé : .../images/foo.jpg → .../labels/foo.txt (s'il existe)
    def labelFor(img: Path): Option[Path] = {
      val candidate = new Path(
        img.toString
          .replace("/images/", "/labels/")
          .replaceAll("(?i)\\.(jpg|jpeg|png)$", ".txt"))
      if (srcFs.exists(candidate)) Some(candidate) else None
    }

    mode match {

      // ─── Flux HEBDOMADAIRE : copie intégrale récursive de ml_dataset ─────────
      case "batch" =>
        val files = listFiles(srcFs, srcQ, recursive)
        println(s"[Producer:batch] ${files.size} fichiers trouvés dans $srcQ (recursive=$recursive)")
        if (files.isEmpty) { println("[Producer:batch] Rien à copier — arrêt."); sc.stop(); return }

        val batches = files.grouped(batchSize).toSeq
        batches.zipWithIndex.foreach { case (batch, idx) =>
          val copied = batch.count(transferRaw(_, move = false))
          println(s"[Producer:batch] Lot ${idx + 1}/${batches.size} : $copied nouveaux fichiers → $destQ")
          if (idx < batches.size - 1 && intervalSeconds > 0)
            Thread.sleep(intervalSeconds * 1000L)
        }
        println(s"[Producer:batch] Terminé — arborescence ml_dataset copiée dans $destQ")

      // ─── Flux CONTINU : veille de continuous_ingress, déplacement au fil de l'eau ─
      case "continuous" =>
        println(s"[Producer:continuous] Veille de $srcQ — lots de $batchSize toutes les ${intervalSeconds}s")
        var running = true
        sys.addShutdownHook { running = false }

        while (running) {
          val images = listFiles(srcFs, srcQ, recursive)
            .filter(_.getName.matches(ImagePattern))
            .take(batchSize)

          if (images.nonEmpty) {
            // Label déplacé AVANT son image : quand le Consumer voit l'image,
            // le label est déjà dans DEST.
            val moved = images.count { img =>
              labelFor(img).foreach(transferRaw(_, move = true))
              transferRaw(img, move = true)
            }
            println(s"[Producer:continuous] $moved images déplacées → $destQ")
          }
          Thread.sleep(math.max(1L, intervalSeconds) * 1000L)
        }
    }

    sc.stop()
  }
}
