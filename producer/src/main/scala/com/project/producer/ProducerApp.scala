package com.project.producer

import com.project.common._
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

/**
 * Producer BATCH (Spark Core + Scala).
 * - Lit les images + annotations (YOLO / COCO / Pascal VOC) du dataset license-plate-detection (train/valid/test)
 * - Letterbox resize (256x256) + reprojection des bounding boxes
 * - Calcule les features bbox (aire, ratio, quadrants, num_boxes...)
 * - Écrit UNE SEULE FOIS en Parquet dans hdfs:///project/raw/
 * - S'arrête après exécution (pas de streaming, pas de boucle)
 */
object ProducerApp {

  private val splits = Seq("train", "valid", "test")

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

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("PlateDetection-Producer-Batch")
      .getOrCreate()

    import spark.implicits._

    // Chemin du dataset (HDFS ou local). Par défaut hdfs:///project/staging/dataset
    // Structure attendue : <datasetPath>/{train,valid,test}/{images,labels}/
    val datasetPath = if (args.nonEmpty) args(0) else HdfsPaths.STAGING + "dataset"
    val imageExts = Seq("jpg", "jpeg", "png", "JPG", "JPEG", "PNG")

    // --- 1. Lecture des images ---
    val imagePaths = for (s <- splits; e <- imageExts) yield s"$datasetPath/$s/images/*.$e"
    val dfImages = spark.read.format("binaryFile")
      .load(imagePaths: _*)
      .withColumn("split", regexp_extract(col("path"), "/(train|valid|test)/images/", 1))
      .withColumn("image_name", regexp_extract(col("path"), "(?i)([^/]+)\\.(jpg|jpeg|png)$", 1))
      .select(
        col("image_name"),
        col("split"),
        col("path").as("image_path"),
        col("content").as("raw_content")
      )

    val nbImages = dfImages.count()
    println(s"Images chargées : $nbImages")

    // --- 2. Lecture et parsing des annotations (YOLO txt, COCO json, Pascal VOC xml) ---
    val classNames = readClassNames(spark, datasetPath)
    val dfLabels = readYoloAnnotations(spark, datasetPath, classNames)
      .unionByName(readCocoAnnotations(spark, datasetPath))
      .unionByName(readVocAnnotations(spark, datasetPath))

    // --- 3. Jointure images <-> labels (left : une image peut avoir 0..N plaques) ---
    val dfJoined = dfImages.join(dfLabels, Seq("image_name", "split"), "left")

    // --- 4. Letterbox resize (256x256) ---
    val dfLetterbox = dfJoined
      .withColumn("lb", ImagePreprocessing.letterboxResize(col("raw_content")))
      .filter(col("lb").isNotNull)
      .withColumn("image_data", col("lb.image"))
      .withColumn("pad_x", col("lb.padX"))
      .withColumn("pad_y", col("lb.padY"))
      .withColumn("orig_width", col("lb.origWidth"))
      .withColumn("orig_height", col("lb.origHeight"))
      .drop("lb", "raw_content")

    // --- 5. Reprojection des bounding boxes sur l'image letterboxée ---
    val regionW = lit(1.0) - lit(2.0) * col("pad_x")
    val regionH = lit(1.0) - lit(2.0) * col("pad_y")

    val dfBoxes = dfLetterbox
      .withColumn("cx_norm", col("cx") * regionW + col("pad_x"))
      .withColumn("cy_norm", col("cy") * regionH + col("pad_y"))
      .withColumn("w_norm", col("w") * regionW)
      .withColumn("h_norm", col("h") * regionH)
      .drop("cx", "cy", "w", "h", "pad_x", "pad_y")

    // --- 6. Features bbox (coordonnées absolues, aires, ratio, quadrants) ---
    val imgSize = lit(PreprocessingConfig.IMG_SIZE.toDouble)
    val dfFeatures = dfBoxes
      .withColumn("x_min", (col("cx_norm") - col("w_norm") / 2) * imgSize)
      .withColumn("y_min", (col("cy_norm") - col("h_norm") / 2) * imgSize)
      .withColumn("x_max", (col("cx_norm") + col("w_norm") / 2) * imgSize)
      .withColumn("y_max", (col("cy_norm") + col("h_norm") / 2) * imgSize)
      .withColumn("bbox_area", (col("x_max") - col("x_min")) * (col("y_max") - col("y_min")))
      .withColumn("bbox_area_norm", col("w_norm") * col("h_norm"))
      .withColumn("aspect_ratio", col("w_norm") / col("h_norm"))
      .withColumn("bbox_size_category", Features.sizeCategoryExpr(col("bbox_area_norm")))
      .withColumn("center_x_quadrant", Features.xQuadrantExpr(col("cx_norm")))
      .withColumn("center_y_quadrant", Features.yQuadrantExpr(col("cy_norm")))

    // --- 7. Nombre de plaques détectées par image ---
    val numBoxesWindow = Window.partitionBy("image_name", "split")
    val dfFinal = dfFeatures
      .withColumn("num_boxes", count(col("class_id")).over(numBoxesWindow).cast(IntegerType))
      .withColumn("ingested_at", current_timestamp())
      .select(Schemas.preprocessedSchema.fieldNames.map(col): _*)

    val nbRows = dfFinal.count()
    println(s"Lignes après prétraitement : $nbRows")

    // --- 8. Écriture unique en Parquet (Snappy) vers HDFS ---
    dfFinal.write
      .mode("overwrite")
      .option("compression", "snappy")
      .parquet(HdfsPaths.RAW)

    println(s"Données écrites dans ${HdfsPaths.RAW}")

    // --- 9. Log d'exécution (hdfs:///project/logs/) ---
    Seq((java.time.Instant.now().toString, datasetPath, nbImages, nbRows))
      .toDF("run_at", "dataset_path", "nb_images", "nb_rows")
      .write.mode("append").parquet(HdfsPaths.LOGS + "producer/")

    println(s"Log d'exécution écrit dans ${HdfsPaths.LOGS}producer/")

    // --- 10. Arrêt (batch unique, pas de boucle) ---
    spark.stop()
  }

  private def emptyAnnotationDf(spark: SparkSession): DataFrame = {
    import spark.implicits._
    Seq.empty[AnnotationRow].toDF()
  }

  // classes.txt optionnel à la racine du dataset (1 nom de classe par ligne, index = class_id), pour les labels YOLO
  private def readClassNames(spark: SparkSession, datasetPath: String): Map[Int, String] = {
    try {
      spark.read.text(s"$datasetPath/classes.txt").collect()
        .map(_.getString(0).trim)
        .zipWithIndex.map { case (name, idx) => idx -> name }.toMap
    } catch {
      case _: Exception => Map.empty
    }
  }

  // --- YOLO : <split>/labels/*.txt, format "class_id cx cy w h" (normalisé) ---
  private def readYoloAnnotations(spark: SparkSession, datasetPath: String, classNames: Map[Int, String]): DataFrame = {
    import spark.implicits._
    val labelPaths = splits.map(s => s"$datasetPath/$s/labels/*.txt")
    try {
      val labelOf = udf((id: Int) => classNames.getOrElse(id, s"class_$id"))
      spark.read.text(labelPaths: _*)
        .withColumn("label_path", input_file_name())
        .withColumn("split", regexp_extract(col("label_path"), "/(train|valid|test)/labels/", 1))
        .withColumn("image_name", regexp_extract(col("label_path"), "([^/]+)\\.txt$", 1))
        .filter(trim(col("value")) =!= "")
        .withColumn("parts", split(trim(col("value")), "\\s+"))
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
    } catch {
      case _: Exception => emptyAnnotationDf(spark)
    }
  }

  // --- COCO : <split>/_annotations.coco.json (images[], annotations[], categories[]), bbox=[x,y,w,h] en pixels ---
  private def readCocoAnnotations(spark: SparkSession, datasetPath: String): DataFrame = {
    val perSplit = splits.flatMap { s =>
      val path = s"$datasetPath/$s/_annotations.coco.json"
      try {
        val raw = spark.read.option("multiLine", value = true).json(path)

        val images = raw.select(explode(col("images")).as("img"))
          .select(
            col("img.id").as("image_id"),
            col("img.file_name").as("file_name"),
            col("img.width").cast(DoubleType).as("img_w"),
            col("img.height").cast(DoubleType).as("img_h")
          )

        val categories = raw.select(explode(col("categories")).as("cat"))
          .select(col("cat.id").as("category_id"), col("cat.name").as("label"))

        val annotations = raw.select(explode(col("annotations")).as("ann"))
          .select(
            col("ann.image_id").as("image_id"),
            col("ann.category_id").as("category_id"),
            col("ann.bbox").as("bbox")
          )

        val joined = annotations.join(images, "image_id").join(categories, "category_id")

        Some(joined.select(
          regexp_extract(col("file_name"), "([^/.]+)\\.[^.]+$", 1).as("image_name"),
          lit(s).as("split"),
          col("category_id").cast(IntegerType).as("class_id"),
          col("label"),
          ((col("bbox").getItem(0) + col("bbox").getItem(2) / 2.0) / col("img_w")).as("cx"),
          ((col("bbox").getItem(1) + col("bbox").getItem(3) / 2.0) / col("img_h")).as("cy"),
          (col("bbox").getItem(2) / col("img_w")).as("w"),
          (col("bbox").getItem(3) / col("img_h")).as("h")
        ))
      } catch {
        case _: Exception => None
      }
    }
    if (perSplit.isEmpty) emptyAnnotationDf(spark) else perSplit.reduce(_.unionByName(_))
  }

  // --- Pascal VOC : <split>/labels/*.xml (<annotation><size>.. <object><name/><bndbox>{xmin,ymin,xmax,ymax}</bndbox></object>) ---
  private def readVocAnnotations(spark: SparkSession, datasetPath: String): DataFrame = {
    val xmlPaths = splits.map(s => s"$datasetPath/$s/labels/*.xml")
    try {
      val parseVoc = udf((content: String) =>
        AnnotationParsers.parseVocXml(content).map(a => (a.classId, a.label, a.cx, a.cy, a.w, a.h))
      )
      spark.read.option("wholetext", value = true).text(xmlPaths: _*)
        .withColumn("ann_path", input_file_name())
        .withColumn("split", regexp_extract(col("ann_path"), "/(train|valid|test)/labels/", 1))
        .withColumn("image_name", regexp_extract(col("ann_path"), "([^/]+)\\.xml$", 1))
        .withColumn("objs", parseVoc(col("value")))
        .filter(size(col("objs")) > 0)
        .withColumn("obj", explode(col("objs")))
        .select(
          col("image_name"),
          col("split"),
          col("obj._1").cast(IntegerType).as("class_id"),
          col("obj._2").as("label"),
          col("obj._3").as("cx"),
          col("obj._4").as("cy"),
          col("obj._5").as("w"),
          col("obj._6").as("h")
        )
    } catch {
      case _: Exception => emptyAnnotationDf(spark)
    }
  }
}
