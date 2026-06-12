import org.apache.spark.sql.{SparkSession, Row}
import org.apache.spark.sql.functions._
import java.sql.Timestamp

/**
 * Producer BATCH (Spark Core + Scala).
 * - Génère / lit un dataset de détections de plaques
 * - Nettoie et transforme
 * - Écrit UNE SEULE FOIS en Parquet dans hdfs:///project/raw/
 * - S'arrête après exécution (pas de streaming, pas de boucle)
 */
object ProducerApp {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("PlateDetection-Producer-Batch")
      .getOrCreate()

    import spark.implicits._

    // --- 1. Lecture du dataset ---
    // Si un chemin CSV est passé en argument, on le lit. Sinon on génère des données simulées.
    val rawDF = if (args.nonEmpty) {
      spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .schema(Schemas.plateDetectionSchema)
        .csv(args(0))
    } else {
      val now = System.currentTimeMillis()
      val data = Seq(
        ("d-0001", "AB-123-CD", "car",   "cam-01", 0.97, new Timestamp(now)),
        ("d-0002", "EF-456-GH", "truck", "cam-02", 0.81, new Timestamp(now)),
        ("d-0003", "IJ-789-KL", "car",   "cam-01", 0.55, new Timestamp(now)),
        ("d-0004", "",          "car",   "cam-03", 0.40, new Timestamp(now)),
        ("d-0005", "MN-321-OP", "moto",  "cam-02", 0.92, new Timestamp(now)),
        ("d-0006", "QR-654-ST", "car",   "cam-04", 0.88, new Timestamp(now)),
        ("d-0007", "UV-987-WX", null,    "cam-01", 0.73, new Timestamp(now)),
        ("d-0008", "AB-123-CD", "car",   "cam-01", 0.97, new Timestamp(now)) // doublon volontaire
      )
      data.toDF("detection_id", "plate_number", "vehicle_type", "camera_id", "confidence", "detected_at")
    }

    // --- 2. Nettoyage et transformation ---
    val cleanedDF = rawDF
      .na.drop(Seq("detection_id", "plate_number"))
      .filter(col("plate_number") =!= "")
      .filter(col("confidence").between(0.0, 1.0))
      .dropDuplicates("detection_id")
      .withColumn("vehicle_type", coalesce(col("vehicle_type"), lit("unknown")))
      .withColumn("plate_number", upper(trim(col("plate_number"))))

    println(s"Lignes après nettoyage : ${cleanedDF.count()}")

    // --- 3. Écriture unique en Parquet vers HDFS ---
    cleanedDF.write
      .mode("overwrite")
      .parquet(HdfsPaths.RAW)

    println(s"Données écrites dans ${HdfsPaths.RAW}")

    // --- 4. Arrêt (batch unique, pas de boucle) ---
    spark.stop()
  }
}
