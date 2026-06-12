import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

/**
 * Producer BATCH (Spark Core + Scala).
 * - Lit les images + labels YOLO du dataset license-plate-detection (train/valid/test)
 * - Letterbox resize (256x256) + reprojection des bounding boxes
 * - Calcule les features bbox (aire, ratio, quadrants, num_boxes...)
 * - Écrit UNE SEULE FOIS en Parquet dans hdfs:///project/raw/
 * - S'arrête après exécution (pas de streaming, pas de boucle)
 */
object ProducerApp {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("PlateDetection-Producer-Batch")
      .getOrCreate()

    import spark.implicits._

    // Chemin du dataset (HDFS ou local). Par défaut hdfs:///project/staging/dataset
    // Structure attendue : <datasetPath>/{train,valid,test}/{images,labels}/
    val datasetPath = if (args.nonEmpty) args(0) else HdfsPaths.STAGING + "dataset"
    val splits = Seq("train", "valid", "test")
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

    // --- 2. Lecture et parsing des labels YOLO (class_id cx cy w h) ---
    val labelPaths = splits.map(s => s"$datasetPath/$s/labels/*.txt")
    val dfLabels = spark.read.text(labelPaths: _*)
      .withColumn("label_path", input_file_name())
      .withColumn("split", regexp_extract(col("label_path"), "/(train|valid|test)/labels/", 1))
      .withColumn("image_name", regexp_extract(col("label_path"), "([^/]+)\\.txt$", 1))
      .filter(trim(col("value")) =!= "")
      .withColumn("parts", split(trim(col("value")), "\\s+"))
      .select(
        col("image_name"),
        col("split"),
        col("parts").getItem(0).cast(IntegerType).as("class_id"),
        col("parts").getItem(1).cast(DoubleType).as("cx"),
        col("parts").getItem(2).cast(DoubleType).as("cy"),
        col("parts").getItem(3).cast(DoubleType).as("w"),
        col("parts").getItem(4).cast(DoubleType).as("h")
      )

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

    // --- 8. Écriture unique en Parquet vers HDFS ---
    dfFinal.write
      .mode("overwrite")
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
}
