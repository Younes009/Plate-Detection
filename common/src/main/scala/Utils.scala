import org.apache.spark.sql.Column
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import java.awt.{Color, RenderingHints}
import java.awt.image.BufferedImage
import java.io.{ByteArrayInputStream, ByteArrayOutputStream}
import javax.imageio.ImageIO

object HdfsPaths {
  val RAW        = "hdfs:///project/raw/"
  val PROCESSED  = "hdfs:///project/processed/"
  val STAGING    = "hdfs:///project/staging/"
  val LOGS       = "hdfs:///project/logs/"
  val CHECKPOINT = "hdfs:///project/checkpoint/"
}

object PreprocessingConfig {
  // Taille cible (carrée) après letterbox resize
  val IMG_SIZE = 256
}

object Schemas {
  // Schema du dataset prétraité plaques (sortie producer / entrée streaming consumer)
  val preprocessedSchema: StructType = StructType(Seq(
    StructField("image_name", StringType, nullable = false),
    StructField("split", StringType, nullable = false),
    StructField("image_path", StringType, nullable = false),
    StructField("orig_width", IntegerType, nullable = false),
    StructField("orig_height", IntegerType, nullable = false),
    StructField("image_data", BinaryType, nullable = false),
    StructField("class_id", IntegerType, nullable = true),
    StructField("cx_norm", DoubleType, nullable = true),
    StructField("cy_norm", DoubleType, nullable = true),
    StructField("w_norm", DoubleType, nullable = true),
    StructField("h_norm", DoubleType, nullable = true),
    StructField("x_min", DoubleType, nullable = true),
    StructField("y_min", DoubleType, nullable = true),
    StructField("x_max", DoubleType, nullable = true),
    StructField("y_max", DoubleType, nullable = true),
    StructField("bbox_area", DoubleType, nullable = true),
    StructField("bbox_area_norm", DoubleType, nullable = true),
    StructField("aspect_ratio", DoubleType, nullable = true),
    StructField("bbox_size_category", StringType, nullable = true),
    StructField("center_x_quadrant", StringType, nullable = true),
    StructField("center_y_quadrant", StringType, nullable = true),
    StructField("num_boxes", IntegerType, nullable = false),
    StructField("ingested_at", TimestampType, nullable = false)
  ))
}

// Résultat du letterbox resize : image encodée JPEG (IMG_SIZE x IMG_SIZE) + infos de reprojection
case class LetterboxResult(
  image: Array[Byte],
  padX: Double,
  padY: Double,
  origWidth: Int,
  origHeight: Int
)

object ImagePreprocessing {
  import PreprocessingConfig.IMG_SIZE

  /**
   * Redimensionne une image en conservant son ratio d'aspect (letterbox) puis ajoute
   * du padding noir pour obtenir une image carrée IMG_SIZE x IMG_SIZE.
   * padX/padY (normalisés [0,1]) permettent de reprojeter les bounding boxes YOLO.
   */
  val letterboxResize: UserDefinedFunction = udf((content: Array[Byte]) => {
    try {
      val src = ImageIO.read(new ByteArrayInputStream(content))
      val origW = src.getWidth
      val origH = src.getHeight

      val scale = math.min(IMG_SIZE.toDouble / origW, IMG_SIZE.toDouble / origH)
      val newW = math.max(1, (origW * scale).toInt)
      val newH = math.max(1, (origH * scale).toInt)

      val padX = (IMG_SIZE - newW) / 2
      val padY = (IMG_SIZE - newH) / 2

      val canvas = new BufferedImage(IMG_SIZE, IMG_SIZE, BufferedImage.TYPE_INT_RGB)
      val g = canvas.createGraphics()
      g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR)
      g.setColor(Color.BLACK)
      g.fillRect(0, 0, IMG_SIZE, IMG_SIZE)
      g.drawImage(src, padX, padY, newW, newH, null)
      g.dispose()

      val out = new ByteArrayOutputStream()
      ImageIO.write(canvas, "jpg", out)

      LetterboxResult(out.toByteArray, padX.toDouble / IMG_SIZE, padY.toDouble / IMG_SIZE, origW, origH)
    } catch {
      case _: Exception => null
    }
  })
}

object Features {
  // Catégorie de taille de bbox selon son aire normalisée (letterboxé)
  def sizeCategoryExpr(areaNorm: Column): Column =
    when(areaNorm.isNull, lit(null: String))
      .when(areaNorm < 0.02, lit("small"))
      .when(areaNorm < 0.1, lit("medium"))
      .otherwise(lit("large"))

  // Quadrant horizontal du centre de la bbox
  def xQuadrantExpr(cxNorm: Column): Column =
    when(cxNorm.isNull, lit(null: String))
      .when(cxNorm < 0.5, lit("left"))
      .otherwise(lit("right"))

  // Quadrant vertical du centre de la bbox
  def yQuadrantExpr(cyNorm: Column): Column =
    when(cyNorm.isNull, lit(null: String))
      .when(cyNorm < 0.5, lit("top"))
      .otherwise(lit("bottom"))
}
