import org.apache.spark.sql.types._

object HdfsPaths {
  val RAW        = "hdfs:///project/raw/"
  val PROCESSED  = "hdfs:///project/processed/"
  val STAGING    = "hdfs:///project/staging/"
  val LOGS       = "hdfs:///project/logs/"
  val CHECKPOINT = "hdfs:///project/checkpoint/"
}

object Schemas {
  // Schema des événements de détection de plaques (entrée raw + streaming)
  val plateDetectionSchema: StructType = StructType(Seq(
    StructField("detection_id", StringType, nullable = false),
    StructField("plate_number", StringType, nullable = false),
    StructField("vehicle_type", StringType, nullable = true),
    StructField("camera_id", StringType, nullable = true),
    StructField("confidence", DoubleType, nullable = true),
    StructField("detected_at", TimestampType, nullable = true)
  ))
}

object Scoring {
  // Score qualité simulé : combine confidence du modèle + bonus longueur plaque valide
  def riskScoreExpr = {
    import org.apache.spark.sql.functions._
    when(col("confidence") >= 0.9, lit("HIGH"))
      .when(col("confidence") >= 0.6, lit("MEDIUM"))
      .otherwise(lit("LOW"))
  }
}
