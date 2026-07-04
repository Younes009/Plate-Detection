package com.project.common

import org.apache.spark.sql.Column
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import java.awt.{Color, RenderingHints}
import java.awt.image.BufferedImage
import java.io.{ByteArrayInputStream, ByteArrayOutputStream}
import javax.imageio.ImageIO
import scala.xml.XML

object HdfsPaths {
  val SOURCE               = "hdfs:///project/source/"
  val DEST                 = "hdfs:///project/dest/"
  val OUTPUT               = "hdfs:///project/output/"
  val OUTPUT_PREDICTIONS   = "hdfs:///project/output/predictions/"
  val MODELS               = "hdfs:///project/models/"
  val CHECKPOINT           = "hdfs:///project/checkpoint/"
  val LOGS                 = "hdfs:///project/logs/"
}

object PreprocessingConfig {
  val IMG_SIZE = 256
}

object Schemas {

  // Schema écrit par le Producer dans DEST/ et lu en streaming par le Consumer
  val destSchema: StructType = StructType(Seq(
    StructField("image_name",          StringType,    nullable = false),
    StructField("split",               StringType,    nullable = false),
    StructField("image_path",          StringType,    nullable = false),
    StructField("label",               StringType,    nullable = true),
    StructField("orig_width",          IntegerType,   nullable = false),
    StructField("orig_height",         IntegerType,   nullable = false),
    StructField("image_data",          BinaryType,    nullable = false),
    StructField("class_id",            IntegerType,   nullable = true),
    StructField("cx_norm",             DoubleType,    nullable = true),
    StructField("cy_norm",             DoubleType,    nullable = true),
    StructField("w_norm",              DoubleType,    nullable = true),
    StructField("h_norm",              DoubleType,    nullable = true),
    StructField("x_min",               DoubleType,    nullable = true),
    StructField("y_min",               DoubleType,    nullable = true),
    StructField("x_max",               DoubleType,    nullable = true),
    StructField("y_max",               DoubleType,    nullable = true),
    StructField("bbox_area",           DoubleType,    nullable = true),
    StructField("bbox_area_norm",      DoubleType,    nullable = true),
    StructField("aspect_ratio",        DoubleType,    nullable = true),
    StructField("bbox_size_category",  StringType,    nullable = true),
    StructField("center_x_quadrant",   StringType,    nullable = true),
    StructField("center_y_quadrant",   StringType,    nullable = true),
    StructField("num_boxes",           IntegerType,   nullable = false),
    StructField("ingested_at",         TimestampType, nullable = false)
  ))

  // Schema écrit par le Consumer dans OUTPUT/ (destSchema + enrichissement + scoring)
  val outputSchema: StructType = destSchema
    .add(StructField("quality_flag",     StringType,  nullable = true))
    .add(StructField("prediction",       StringType,  nullable = true))
    .add(StructField("confidence_score", DoubleType,  nullable = true))
    .add(StructField("processed_at",     TimestampType, nullable = false))

  // Alias rétrocompat
  val preprocessedSchema: StructType = destSchema
}

case class LetterboxResult(
  image: Array[Byte],
  padX: Double,
  padY: Double,
  origWidth: Int,
  origHeight: Int
)

object ImagePreprocessing {
  import PreprocessingConfig.IMG_SIZE

  val letterboxResize: UserDefinedFunction = udf((content: Array[Byte]) => {
    try {
      val src  = ImageIO.read(new ByteArrayInputStream(content))
      val origW = src.getWidth
      val origH = src.getHeight
      val scale = math.min(IMG_SIZE.toDouble / origW, IMG_SIZE.toDouble / origH)
      val newW  = math.max(1, (origW * scale).toInt)
      val newH  = math.max(1, (origH * scale).toInt)
      val padX  = (IMG_SIZE - newW) / 2
      val padY  = (IMG_SIZE - newH) / 2
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
  def sizeCategoryExpr(areaNorm: Column): Column =
    when(areaNorm.isNull, lit(null: String))
      .when(areaNorm < 0.02, lit("small"))
      .when(areaNorm < 0.1,  lit("medium"))
      .otherwise(lit("large"))

  def xQuadrantExpr(cxNorm: Column): Column =
    when(cxNorm.isNull, lit(null: String))
      .when(cxNorm < 0.5,  lit("left"))
      .otherwise(lit("right"))

  def yQuadrantExpr(cyNorm: Column): Column =
    when(cyNorm.isNull, lit(null: String))
      .when(cyNorm < 0.5,  lit("top"))
      .otherwise(lit("bottom"))
}

case class RawAnnotation(classId: Int, label: String, cx: Double, cy: Double, w: Double, h: Double)

object AnnotationParsers {

  def parseYoloLine(line: String, classNames: Map[Int, String]): Option[RawAnnotation] = {
    val parts = line.trim.split("\\s+")
    if (parts.length < 5) None
    else {
      val classId = parts(0).toInt
      Some(RawAnnotation(
        classId,
        classNames.getOrElse(classId, s"class_$classId"),
        parts(1).toDouble, parts(2).toDouble, parts(3).toDouble, parts(4).toDouble
      ))
    }
  }

  def parseVocXml(content: String): Seq[RawAnnotation] = {
    try {
      val xml  = XML.loadString(content)
      val imgW = (xml \ "size" \ "width").text.trim.toDouble
      val imgH = (xml \ "size" \ "height").text.trim.toDouble
      (xml \ "object").map { obj =>
        val name = (obj \ "name").text.trim
        val xmin = (obj \ "bndbox" \ "xmin").text.trim.toDouble
        val ymin = (obj \ "bndbox" \ "ymin").text.trim.toDouble
        val xmax = (obj \ "bndbox" \ "xmax").text.trim.toDouble
        val ymax = (obj \ "bndbox" \ "ymax").text.trim.toDouble
        val w  = (xmax - xmin) / imgW
        val h  = (ymax - ymin) / imgH
        val cx = (xmin + xmax) / 2.0 / imgW
        val cy = (ymin + ymax) / 2.0 / imgH
        RawAnnotation(math.abs(name.hashCode) % 1000, name, cx, cy, w, h)
      }
    } catch {
      case _: Exception => Seq.empty
    }
  }
}
