package com.project.producer

import com.project.common._
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

/**
 * Microservice 1 — Producer BATCH (Spark Core + Scala).
 *
 * Args : [sourcePath] [destPath] [batchSize] [intervalSeconds]
 *
 * Logique :
 *   1. Crée sourcePath s'il n'existe pas
 *   2. Parcourt récursivement sourcePath pour lister toutes les images (JPEG/PNG)
 *      et tous les fichiers d'annotations (.txt YOLO / .json COCO / .xml VOC)
 *   3. Charge les annotations UNE SEULE FOIS (cache Spark)
 *   4. Découpe la liste d'images en batches de batchSize
 *   5. Pour chaque batch :
 *        - Lit les images (binaryFile) depuis la liste de chemins du batch
 *        - Jointure avec les annotations
 *        - Letterbox 256×256 + reprojection bbox + calcul features
 *        - Écrit le résultat en Parquet (Snappy) dans destPath/batch_N_TS/
 *        - Attend intervalSeconds avant le batch suivant
 *   6. Écrit un log d'exécution dans LOGS/producer/
 *   7. Arrêt automatique (batch pur, aucune boucle infinie)
 */
object ProducerApp {

  private case class AnnotationRow(
    image_name: String,
    split: String,
    class_id: Option[Int],
    label: Option[String],
    cx: Option[Double],
    cy: Option[Double],
    w: Option[Double],
    h: Option[Double]
  )

  private def extractSplitExpr(pathCol: org.apache.spark.sql.Column): org.apache.spark.sql.Column =
    when(
      length(regexp_extract(pathCol, "/(train|valid|test)/", 1)) > 0,
      regexp_extract(pathCol, "/(train|valid|test)/", 1)
    ).otherwise(lit("unknown"))

  private def extractSplitStr(path: String): String =
    "/(train|valid|test)/".r.findFirstMatchIn(path).map(_.group(1)).getOrElse("unknown")

  // Liste récursive via Hadoop FileSystem (HDFS ou local)
  private def listFilesRecursively(fs: FileSystem, path: Path, predicate: String => Boolean): Seq[String] =
    try {
      fs.listStatus(path).flatMap { s =>
        if (s.isDirectory) listFilesRecursively(fs, s.getPath, predicate)
        else if (predicate(s.getPath.getName)) Seq(s.getPath.toString)
        else Seq.empty
      }
    } catch { case _: Exception => Seq.empty }

  // Crée le dossier si absent, retourne le FileSystem
  private def ensureDirectory(spark: SparkSession, pathStr: String): FileSystem = {
    val p  = new Path(pathStr)
    val fs = p.getFileSystem(spark.sparkContext.hadoopConfiguration)
    if (!fs.exists(p)) {
      fs.mkdirs(p)
      println(s"[Producer] Dossier créé : $pathStr")
    }
    fs
  }

  def main(args: Array[String]): Unit = {
    val sourcePath      = if (args.length > 0) args(0) else HdfsPaths.SOURCE
    val destPath        = if (args.length > 1) args(1) else HdfsPaths.DEST
    val batchSize       = if (args.length > 2) args(2).toInt  else 100
    val intervalSeconds = if (args.length > 3) args(3).toLong else 30L

    val spark = SparkSession.builder()
      .appName("PlateDetection-Producer-Batch")
      .getOrCreate()

    import spark.implicits._

    val fs = ensureDirectory(spark, sourcePath)
    ensureDirectory(spark, destPath)

    // --- 1. Liste récursive de toutes les images ---
    val allImages = listFilesRecursively(fs, new Path(sourcePath),
      name => name.toLowerCase.matches(".*\\.(jpg|jpeg|png)$"))

    println(s"[Producer] ${allImages.size} images trouvées dans $sourcePath")
    if (allImages.isEmpty) {
      println("[Producer] Aucune image à traiter, arrêt.")
      spark.stop()
      return
    }

    // --- 2. Chargement unique des annotations (cache) ---
    val classNames   = readClassNames(spark, fs, sourcePath)
    val dfAnnotations = readYoloAnnotations(spark, fs, sourcePath, classNames)
      .unionByName(readCocoAnnotations(spark, fs, sourcePath))
      .unionByName(readVocAnnotations(spark, fs, sourcePath))
      .cache()

    val nbAnnotations = dfAnnotations.count()
    println(s"[Producer] $nbAnnotations annotations chargées")

    // --- 3. Traitement par batches ---
    val batches  = allImages.grouped(batchSize).toSeq
    var totalRows = 0L

    batches.zipWithIndex.foreach { case (batchPaths, batchIdx) =>
      val batchTs = System.currentTimeMillis()
      println(s"[Producer] Batch $batchIdx/${batches.size - 1} : ${batchPaths.size} images")

      val batchDest = s"$destPath/batch_${batchIdx}_$batchTs"
      val nbRows = processBatch(spark, batchPaths, dfAnnotations, batchDest)
      totalRows += nbRows
      println(s"[Producer] Batch $batchIdx écrit dans $batchDest ($nbRows lignes)")

      if (batchIdx < batches.size - 1 && intervalSeconds > 0) {
        println(s"[Producer] Attente ${intervalSeconds}s avant le prochain batch...")
        Thread.sleep(intervalSeconds * 1000L)
      }
    }

    // --- 4. Log d'exécution ---
    Seq((java.time.Instant.now().toString, sourcePath, destPath, allImages.size.toLong, batches.size, totalRows))
      .toDF("run_at", "source_path", "dest_path", "nb_images", "nb_batches", "nb_rows")
      .write.mode("append").parquet(HdfsPaths.LOGS + "producer/")

    println(s"[Producer] Terminé — $totalRows lignes dans $destPath")
    spark.stop()
  }

  // Traite un batch de chemins d'images, écrit en Parquet, retourne le nombre de lignes
  private def processBatch(
    spark: SparkSession,
    imagePaths: Seq[String],
    dfAnnotations: DataFrame,
    destDir: String
  ): Long = {
    import spark.implicits._

    // Lecture des images du batch (chemins explicites)
    val dfImages = spark.read.format("binaryFile")
      .load(imagePaths: _*)
      .withColumn("split", extractSplitExpr(col("path")))
      .withColumn("image_name", regexp_extract(col("path"), "(?i)([^/]+)\\.(jpg|jpeg|png)$", 1))
      .select(
        col("image_name"),
        col("split"),
        col("path").as("image_path"),
        col("content").as("raw_content")
      )

    // Jointure avec les annotations (left : image sans annotation = background)
    val dfJoined = dfImages.join(dfAnnotations, Seq("image_name", "split"), "left")

    // Letterbox 256×256
    val dfLetterbox = dfJoined
      .withColumn("lb", ImagePreprocessing.letterboxResize(col("raw_content")))
      .filter(col("lb").isNotNull)
      .withColumn("image_data",   col("lb.image"))
      .withColumn("pad_x",        col("lb.padX"))
      .withColumn("pad_y",        col("lb.padY"))
      .withColumn("orig_width",   col("lb.origWidth"))
      .withColumn("orig_height",  col("lb.origHeight"))
      .drop("lb", "raw_content")

    // Reprojection bbox sur image letterboxée
    val regionW = lit(1.0) - lit(2.0) * col("pad_x")
    val regionH = lit(1.0) - lit(2.0) * col("pad_y")
    val dfBoxes = dfLetterbox
      .withColumn("cx_norm", col("cx") * regionW + col("pad_x"))
      .withColumn("cy_norm", col("cy") * regionH + col("pad_y"))
      .withColumn("w_norm",  col("w")  * regionW)
      .withColumn("h_norm",  col("h")  * regionH)
      .drop("cx", "cy", "w", "h", "pad_x", "pad_y")

    // Features bbox (coordonnées absolues, aires, ratio, quadrants)
    val imgSize = lit(PreprocessingConfig.IMG_SIZE.toDouble)
    val dfFeatures = dfBoxes
      .withColumn("x_min",  (col("cx_norm") - col("w_norm") / 2) * imgSize)
      .withColumn("y_min",  (col("cy_norm") - col("h_norm") / 2) * imgSize)
      .withColumn("x_max",  (col("cx_norm") + col("w_norm") / 2) * imgSize)
      .withColumn("y_max",  (col("cy_norm") + col("h_norm") / 2) * imgSize)
      .withColumn("bbox_area",          (col("x_max") - col("x_min")) * (col("y_max") - col("y_min")))
      .withColumn("bbox_area_norm",     col("w_norm") * col("h_norm"))
      .withColumn("aspect_ratio",       col("w_norm") / col("h_norm"))
      .withColumn("bbox_size_category", Features.sizeCategoryExpr(col("bbox_area_norm")))
      .withColumn("center_x_quadrant",  Features.xQuadrantExpr(col("cx_norm")))
      .withColumn("center_y_quadrant",  Features.yQuadrantExpr(col("cy_norm")))

    // Nombre de plaques détectées par image
    val numBoxesWindow = Window.partitionBy("image_name", "split")
    val dfFinal = dfFeatures
      .withColumn("num_boxes",   count(col("class_id")).over(numBoxesWindow).cast(IntegerType))
      .withColumn("ingested_at", current_timestamp())
      .select(Schemas.destSchema.fieldNames.map(col): _*)

    dfFinal.write
      .mode("overwrite")
      .option("compression", "snappy")
      .parquet(destDir)

    dfFinal.count()
  }

  // ─── Helpers annotations ────────────────────────────────────────────────────

  private def emptyAnnotationDf(spark: SparkSession): DataFrame = {
    import spark.implicits._
    Seq.empty[AnnotationRow].toDF()
  }

  private def readClassNames(spark: SparkSession, fs: FileSystem, sourcePath: String): Map[Int, String] = {
    val classFiles = listFilesRecursively(fs, new Path(sourcePath),
      name => name == "classes.txt" || name.endsWith(".names"))
    classFiles.headOption.map { path =>
      try {
        spark.read.text(path).collect()
          .map(_.getString(0).trim)
          .filter(_.nonEmpty)
          .zipWithIndex
          .map { case (name, idx) => idx -> name }
          .toMap
      } catch { case _: Exception => Map.empty[Int, String] }
    }.getOrElse(Map.empty)
  }

  private def readYoloAnnotations(
    spark: SparkSession,
    fs: FileSystem,
    sourcePath: String,
    classNames: Map[Int, String]
  ): DataFrame = {
    import spark.implicits._
    val txtPaths = listFilesRecursively(fs, new Path(sourcePath),
      name => name.toLowerCase.endsWith(".txt") && name != "classes.txt" && !name.endsWith(".names"))
    if (txtPaths.isEmpty) return emptyAnnotationDf(spark)
    try {
      val labelOf = udf((id: Int) => classNames.getOrElse(id, s"class_$id"))
      spark.read.text(txtPaths: _*)
        .withColumn("label_path", input_file_name())
        .withColumn("split",      extractSplitExpr(col("label_path")))
        .withColumn("image_name", regexp_extract(col("label_path"), "([^/]+)\\.txt$", 1))
        .filter(trim(col("value")) =!= "")
        .withColumn("parts", split(trim(col("value")), "\\s+"))
        .filter(size(col("parts")) >= 5)
        .select(
          col("image_name"),
          col("split"),
          col("parts").getItem(0).cast(IntegerType).as("class_id"),
          labelOf(col("parts").getItem(0).cast(IntegerType)).as("label"),
          col("parts").getItem(1).cast(DoubleType).as("cx"),
          col("parts").getItem(2).cast(DoubleType).as("cy"),
          col("parts").getItem(3).cast(DoubleType).as("w"),
          col("parts").getItem(4).cast(DoubleType).as("h")
        )
    } catch { case _: Exception => emptyAnnotationDf(spark) }
  }

  private def readCocoAnnotations(spark: SparkSession, fs: FileSystem, sourcePath: String): DataFrame = {
    val jsonPaths = listFilesRecursively(fs, new Path(sourcePath),
      name => name.toLowerCase.endsWith(".json"))
    val dfs = jsonPaths.flatMap { path =>
      try {
        val raw = spark.read.option("multiLine", value = true).json(path)
        if (!raw.columns.contains("images") || !raw.columns.contains("annotations")) None
        else {
          val splitVal = extractSplitStr(path)
          val images = raw.select(explode(col("images")).as("img"))
            .select(col("img.id").as("image_id"), col("img.file_name").as("file_name"),
                    col("img.width").cast(DoubleType).as("img_w"), col("img.height").cast(DoubleType).as("img_h"))
          val categories = raw.select(explode(col("categories")).as("cat"))
            .select(col("cat.id").as("category_id"), col("cat.name").as("label"))
          val annotations = raw.select(explode(col("annotations")).as("ann"))
            .select(col("ann.image_id").as("image_id"), col("ann.category_id").as("category_id"),
                    col("ann.bbox").as("bbox"))
          val joined = annotations.join(images, "image_id").join(categories, "category_id")
          Some(joined.select(
            regexp_extract(col("file_name"), "([^/.]+)\\.[^.]+$", 1).as("image_name"),
            lit(splitVal).as("split"),
            col("category_id").cast(IntegerType).as("class_id"),
            col("label"),
            ((col("bbox").getItem(0) + col("bbox").getItem(2) / 2.0) / col("img_w")).as("cx"),
            ((col("bbox").getItem(1) + col("bbox").getItem(3) / 2.0) / col("img_h")).as("cy"),
            (col("bbox").getItem(2) / col("img_w")).as("w"),
            (col("bbox").getItem(3) / col("img_h")).as("h")
          ))
        }
      } catch { case _: Exception => None }
    }
    if (dfs.isEmpty) emptyAnnotationDf(spark) else dfs.reduce(_.unionByName(_))
  }

  private def readVocAnnotations(spark: SparkSession, fs: FileSystem, sourcePath: String): DataFrame = {
    val xmlPaths = listFilesRecursively(fs, new Path(sourcePath),
      name => name.toLowerCase.endsWith(".xml"))
    if (xmlPaths.isEmpty) return emptyAnnotationDf(spark)
    try {
      val parseVoc = udf((content: String) =>
        AnnotationParsers.parseVocXml(content).map(a => (a.classId, a.label, a.cx, a.cy, a.w, a.h)))
      spark.read.option("wholetext", value = true).text(xmlPaths: _*)
        .withColumn("ann_path",   input_file_name())
        .withColumn("split",      extractSplitExpr(col("ann_path")))
        .withColumn("image_name", regexp_extract(col("ann_path"), "([^/]+)\\.xml$", 1))
        .withColumn("objs", parseVoc(col("value")))
        .filter(size(col("objs")) > 0)
        .withColumn("obj", explode(col("objs")))
        .select(
          col("image_name"), col("split"),
          col("obj._1").cast(IntegerType).as("class_id"),
          col("obj._2").as("label"),
          col("obj._3").as("cx"),
          col("obj._4").as("cy"),
          col("obj._5").as("w"),
          col("obj._6").as("h")
        )
    } catch { case _: Exception => emptyAnnotationDf(spark) }
  }
}
