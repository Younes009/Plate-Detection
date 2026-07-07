package com.project.ml

import com.project.common.HdfsPaths
import org.apache.hadoop.fs.Path
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

import java.awt.image.BufferedImage
import java.io.ByteArrayInputStream
import javax.imageio.ImageIO
import scala.util.Try

/**
 * Loads and applies the YOLO plate detector to the image stream. Scoring
 * works DIRECTLY on image bytes — the inference stream carries no labels.
 *
 *  - plate_locator.onnx  YOLOv8n exported by ml/train_yolo.py
 *                        input  [1, 3, S, S] RGB /255 CHW
 *                        output [1, 4+k, N] (pixels of S + per-class scores)
 *  - no model file       neutral fallback (uncertain, 0.5)
 *
 * prediction/confidence come from the best detection score; the bbox is the
 * best-scoring box, normalized. ONNX bytes are broadcast once from the
 * driver (initialize()); each partition opens one session, closed by a
 * TaskCompletionListener.
 */
object ModelLoader {

  private val YOLO_ONNX_FILE = "plate_locator.onnx"

  // best score >= threshold -> "plate" (+ bbox), below -> "background"
  private val PLATE_CONF_THRESHOLD = 0.5

  @transient private var broadcastYoloOnnx: Option[Broadcast[Array[Byte]]] = None

  def initialize(spark: SparkSession, modelPath: String = HdfsPaths.MODELS): Unit = {
    val base = modelPath.stripSuffix("/").stripSuffix("\\") + java.io.File.separator
    broadcastYoloOnnx = tryBroadcast(spark, base + YOLO_ONNX_FILE)

    val mode = if (broadcastYoloOnnx.isDefined) s"YOLO ONNX ($YOLO_ONNX_FILE)"
               else "neutral fallback (no model in MODEL/)"
    println(s"[ModelLoader] Active mode: $mode")
  }

  def score(df: DataFrame): DataFrame = broadcastYoloOnnx match {
    case Some(bc) => scoreWithYolo(df, bc)
    case None     => scoreFallback(df)
  }

  // ─── Loading ────────────────────────────────────────────────────────────────

  private def tryBroadcast(spark: SparkSession, filePath: String): Option[Broadcast[Array[Byte]]] =
    Try {
      val p  = new Path(filePath)
      val fs = p.getFileSystem(spark.sparkContext.hadoopConfiguration)
      if (!fs.exists(p)) throw new RuntimeException(s"not found: $filePath")
      val len = fs.getFileStatus(p).getLen.toInt
      val buf = new Array[Byte](len)
      val in  = fs.open(p)
      in.readFully(buf)
      in.close()
      println(s"[ModelLoader] Loaded: $filePath (${len / 1024} KB)")
      spark.sparkContext.broadcast(buf)
    }.toOption

  // SAME column order on every scoring path — divergent Parquet otherwise
  private def withScoringSchema(df: DataFrame): StructType = df.schema
    .add(StructField("prediction",       StringType, nullable = true))
    .add(StructField("confidence_score", DoubleType, nullable = true))
    .add(StructField("bbox_cx",          DoubleType, nullable = true))
    .add(StructField("bbox_cy",          DoubleType, nullable = true))
    .add(StructField("bbox_w",           DoubleType, nullable = true))
    .add(StructField("bbox_h",           DoubleType, nullable = true))

  private val NoBbox: Seq[Any] = Seq(null, null, null, null)

  /** Open an ONNX session for this partition and guarantee it is closed at
   *  task end — success OR failure. Closing only after full iterator
   *  consumption would leak native memory on mid-partition exceptions.
   *  OrtEnvironment is a process singleton and must NOT be closed here. */
  private def newSessionClosedOnTaskEnd(env: ai.onnxruntime.OrtEnvironment,
                                        modelBytes: Array[Byte]): ai.onnxruntime.OrtSession = {
    val session = env.createSession(modelBytes, new ai.onnxruntime.OrtSession.SessionOptions())
    Option(org.apache.spark.TaskContext.get()).foreach(
      _.addTaskCompletionListener[Unit](_ => Try(session.close())))
    session
  }

  /**
   * YOLO raw output [1, 4+k, N] -> (normalized bbox, best score).
   * Rows 0-3 are cx, cy, w, h in PIXELS of the input side; rows 4.. are
   * per-class scores. Best-scoring prediction wins (top-1: one plate per
   * image). Returns (NoBbox, 0.0) on an unexpected format.
   */
  private[ml] def bestDetection(raw: AnyRef, side: Int): (Seq[Any], Double) = {
    def clamp01(v: Double): Any = math.max(0.0, math.min(1.0, v))
    raw match {
      case a3: Array[Array[Array[Float]]] if a3.nonEmpty && a3(0).length >= 5 =>
        val ch = a3(0)
        val n  = ch(0).length
        var bestJ = 0
        var bestS = Float.MinValue
        var j = 0
        while (j < n) {
          var s = Float.MinValue
          var r = 4
          while (r < ch.length) { if (ch(r)(j) > s) s = ch(r)(j); r += 1 }
          if (s > bestS) { bestS = s; bestJ = j }
          j += 1
        }
        val bbox = Seq(ch(0)(bestJ), ch(1)(bestJ), ch(2)(bestJ), ch(3)(bestJ))
          .map(v => clamp01(v.toDouble / side))
        (bbox, math.max(0.0, math.min(1.0, bestS.toDouble)))
      case _ => (NoBbox, 0.0)
    }
  }

  // ─── Scoring: YOLO detector ─────────────────────────────────────────────────

  private def scoreWithYolo(df: DataFrame, bc: Broadcast[Array[Byte]]): DataFrame = {
    val spark        = df.sparkSession
    val outputSchema = withScoringSchema(df)

    val rdd = df.rdd.mapPartitions { rows =>
      val env     = ai.onnxruntime.OrtEnvironment.getEnvironment()
      val session = newSessionClosedOnTaskEnd(env, bc.value)

      // input side deduced from the model: [1, 3, S, S]
      val inputName  = session.getInputInfo.keySet.iterator.next
      val inputShape = session.getInputInfo.values.iterator.next
        .getInfo.asInstanceOf[ai.onnxruntime.TensorInfo].getShape
      require(inputShape.length == 4,
        s"[ModelLoader] unexpected input rank: ${inputShape.mkString("x")} (want [1,3,S,S])")
      val side = inputShape(2).toInt

      rows.map { row =>
        val (pred, conf, bbox) = Try {
          val imgBytes = row.getAs[Array[Byte]]("image_data")
          if (imgBytes == null || imgBytes.isEmpty) ("uncertain", 0.0, NoBbox)
          else {
            val decoded = ImageIO.read(new ByteArrayInputStream(imgBytes))
            val flat    = toFloatCHW(resize(decoded, side))
            // try/finally: native ORT buffers freed even if run() throws
            var tensor: ai.onnxruntime.OnnxTensor        = null
            var result: ai.onnxruntime.OrtSession.Result = null
            try {
              tensor = ai.onnxruntime.OnnxTensor.createTensor(
                env, java.nio.FloatBuffer.wrap(flat),
                Array(1L, 3L, side.toLong, side.toLong))
              val inputs = new java.util.HashMap[String, ai.onnxruntime.OnnxTensorLike]()
              inputs.put(inputName, tensor)

              result = session.run(inputs)
              val (box, score) = bestDetection(result.get(0).getValue, side)
              if (score >= PLATE_CONF_THRESHOLD) ("plate", score, box)
              else ("background", score, NoBbox)
            } finally {
              if (result != null) result.close()
              if (tensor != null) tensor.close()
            }
          }
        }.getOrElse(("uncertain", 0.0, NoBbox))

        Row.fromSeq(row.toSeq ++ Seq(pred, conf) ++ bbox)
      }
    }
    spark.createDataFrame(rdd, outputSchema)
  }

  // ─── Shared image preprocessing ─────────────────────────────────────────────

  /** Resize to S×S RGB, bilinear — must match PIL BILINEAR on the training
   *  side. private[ml] for the Py<->Scala parity test. */
  private[ml] def resize(src: BufferedImage, side: Int): BufferedImage = {
    val out = new BufferedImage(side, side, BufferedImage.TYPE_INT_RGB)
    val g   = out.createGraphics()
    g.setRenderingHint(java.awt.RenderingHints.KEY_INTERPOLATION,
                       java.awt.RenderingHints.VALUE_INTERPOLATION_BILINEAR)
    g.drawImage(src, 0, 0, side, side, null)
    g.dispose()
    out
  }

  /** Image -> flat CHW float32 /255. Bulk getRGB: per-pixel getRGB(x,y)
   *  costs 3*h*w calls on the scoring hot path. */
  private[ml] def toFloatCHW(img: BufferedImage): Array[Float] = {
    val w      = img.getWidth
    val h      = img.getHeight
    val n      = h * w
    val pixels = img.getRGB(0, 0, w, h, null, 0, w)
    val flat   = new Array[Float](3 * n)
    var i = 0
    while (i < n) {
      val rgb = pixels(i)
      flat(i)         = ((rgb >> 16) & 0xFF) / 255.0f
      flat(n + i)     = ((rgb >> 8)  & 0xFF) / 255.0f
      flat(2 * n + i) = ( rgb        & 0xFF) / 255.0f
      i += 1
    }
    flat
  }

  // ─── Neutral fallback (no model available) ──────────────────────────────────

  // same column order as the YOLO path and Schemas.outputSchema
  def scoreFallback(df: DataFrame): DataFrame =
    df.withColumn("prediction", lit("uncertain"))
      .withColumn("confidence_score", lit(0.5))
      .withColumn("bbox_cx", lit(null).cast(DoubleType))
      .withColumn("bbox_cy", lit(null).cast(DoubleType))
      .withColumn("bbox_w",  lit(null).cast(DoubleType))
      .withColumn("bbox_h",  lit(null).cast(DoubleType))
}
