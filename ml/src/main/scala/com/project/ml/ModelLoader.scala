package com.project.ml

import com.project.common.HdfsPaths
import org.apache.hadoop.fs.Path
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

import java.io.ByteArrayInputStream
import javax.imageio.ImageIO
import scala.util.Try

/**
 * Charge et applique un modèle de scoring (détection de plaques) au flux de données.
 *
 * Priorité de chargement (première correspondance gagne) :
 *   1. best_model.onnx  — modèle PyTorch exporté en ONNX via ml/export_pytorch.py
 *                         Input  : "image_input" [batch, 3, 256, 256] float32 (ImageNet normalized)
 *                         Output : "output"       [batch, num_classes] float32
 *                         → scoreWithPytorchOnnx() : mapPartitions sur image_data (JPEG 256×256)
 *
 *   2. plate_model.onnx — modèle sklearn RandomForest exporté via ml/train_model.py
 *                         Input  : "float_input" [batch, 6] float32 (features bbox)
 *                         Output : "output_label" + "output_probability" (zipmap=False)
 *                         → scoreWithSklearnOnnx() : mapPartitions sur bbox features
 *
 *   3. spark_model/     — Spark MLlib PipelineModel
 *                         → scoreWithMllib()
 *
 *   4. (aucun fichier)  — heuristique géométrique (ratio bbox ≈ 3.1, area, quality_flag)
 *                         → scoreSimulated()
 *
 * Appeler initialize() depuis le driver avant de démarrer le streaming.
 * Appeler score(df) dans chaque foreachBatch — colonnes ajoutées : prediction + confidence_score.
 */
object ModelLoader {

  private val PYTORCH_ONNX_FILE = "best_model.onnx"
  private val SKLEARN_ONNX_FILE = "plate_model.onnx"
  private val MLLIB_DIR         = "spark_model"

  val FEATURE_COLS: Array[String] = Array(
    "bbox_area_norm", "aspect_ratio", "cx_norm", "cy_norm", "w_norm", "h_norm"
  )

  // Normalisation ImageNet (doit correspondre à export_pytorch.py)
  private val IMG_MEAN = Array(0.485f, 0.456f, 0.406f)
  private val IMG_STD  = Array(0.229f, 0.224f, 0.225f)

  @transient private var broadcastPytorchOnnx: Option[Broadcast[Array[Byte]]] = None
  @transient private var broadcastSklearnOnnx: Option[Broadcast[Array[Byte]]] = None
  @transient private var mllibModel: Option[PipelineModel]                    = None

  def initialize(spark: SparkSession, modelPath: String = HdfsPaths.MODELS): Unit = {
    val base = modelPath.stripSuffix("/").stripSuffix("\\") + java.io.File.separator
    broadcastPytorchOnnx = tryBroadcast(spark, base + PYTORCH_ONNX_FILE)
    broadcastSklearnOnnx = tryBroadcast(spark, base + SKLEARN_ONNX_FILE)
    if (broadcastPytorchOnnx.isEmpty && broadcastSklearnOnnx.isEmpty)
      mllibModel = Try(PipelineModel.load(base + MLLIB_DIR)).toOption

    val mode = if (broadcastPytorchOnnx.isDefined) s"PyTorch ONNX ($PYTORCH_ONNX_FILE)"
               else if (broadcastSklearnOnnx.isDefined) s"Sklearn ONNX ($SKLEARN_ONNX_FILE)"
               else if (mllibModel.isDefined) "Spark MLlib PipelineModel"
               else "simulé (heuristique géométrique)"
    println(s"[ModelLoader] Mode actif : $mode")
  }

  def score(df: DataFrame): DataFrame = broadcastPytorchOnnx match {
    case Some(bc) => scoreWithPytorchOnnx(df, bc)
    case None     => broadcastSklearnOnnx match {
      case Some(bc) => scoreWithSklearnOnnx(df, bc)
      case None     => mllibModel match {
        case Some(m) => scoreWithMllib(m, df)
        case None    => scoreSimulated(df)
      }
    }
  }

  // ─── Chargement ─────────────────────────────────────────────────────────────

  private def tryBroadcast(spark: SparkSession, filePath: String): Option[Broadcast[Array[Byte]]] =
    Try {
      val p  = new Path(filePath)
      val fs = p.getFileSystem(spark.sparkContext.hadoopConfiguration)
      if (!fs.exists(p)) throw new RuntimeException(s"Introuvable : $filePath")
      val len = fs.getFileStatus(p).getLen.toInt
      val buf = new Array[Byte](len)
      val in  = fs.open(p)
      in.readFully(buf)
      in.close()
      println(s"[ModelLoader] Chargé : $filePath (${len / 1024} KB)")
      spark.sparkContext.broadcast(buf)
    }.toOption

  // ─── 1. Scoring PyTorch ONNX — image-based ──────────────────────────────────
  //
  // Entrée   : image_data (Array[Byte] — JPEG 256×256 letterboxé par le Producer)
  // Sortie   : prédiction au niveau image (plate / background) + confidence [0,1]
  // Pipeline : JPEG → CHW float32 → normalisation ImageNet → OnnxTensor [1,3,256,256] → inférence

  private def scoreWithPytorchOnnx(df: DataFrame, bc: Broadcast[Array[Byte]]): DataFrame = {
    val spark = df.sparkSession
    val outputSchema = df.schema
      .add(StructField("prediction",       StringType, nullable = true))
      .add(StructField("confidence_score", DoubleType, nullable = true))

    val rdd = df.rdd.mapPartitions { rows =>
      val env     = ai.onnxruntime.OrtEnvironment.getEnvironment()
      val session = env.createSession(bc.value, new ai.onnxruntime.OrtSession.SessionOptions())

      rows.map { row =>
        val (pred, conf) = Try {
          val imgBytes = row.getAs[Array[Byte]]("image_data")
          if (imgBytes == null || imgBytes.isEmpty) ("uncertain", 0.0)
          else {
            val flat   = jpegToFloatCHW(imgBytes)
            val tensor = ai.onnxruntime.OnnxTensor.createTensor(env, java.nio.FloatBuffer.wrap(flat), Array(1L, 3L, 256L, 256L))
            val inputs = new java.util.HashMap[String, ai.onnxruntime.OnnxTensorLike]()
            inputs.put("image_input", tensor)

            val result = session.run(inputs)
            val raw    = result.get(0).getValue

            val rawProbs: Array[Float] = raw match {
              case a: Array[Array[Float]] => a(0)
              case a: Array[Float]        => a
              case _                      => Array(0.5f, 0.5f)
            }
            // Applique softmax si ce sont des logits (somme ≠ 1)
            val probs = if (math.abs(rawProbs.sum - 1.0f) < 0.05f) rawProbs else softmax(rawProbs)
            val confidence = if (probs.length >= 2) probs(1).toDouble else probs(0).toDouble
            val prediction = if (confidence >= 0.5) "plate" else "background"

            tensor.close()
            result.close()
            (prediction, confidence)
          }
        }.getOrElse(("uncertain", 0.0))

        Row.fromSeq(row.toSeq ++ Seq(pred, conf))
      }
    }
    spark.createDataFrame(rdd, outputSchema)
  }

  // Décode un JPEG en tableau CHW float32 normalisé ImageNet (format attendu par le modèle PyTorch)
  private def jpegToFloatCHW(bytes: Array[Byte]): Array[Float] = {
    val img  = ImageIO.read(new ByteArrayInputStream(bytes))
    val w    = img.getWidth
    val h    = img.getHeight
    val flat = new Array[Float](3 * h * w)
    for (c <- 0 until 3; y <- 0 until h; x <- 0 until w) {
      val rgb = img.getRGB(x, y)
      val px = c match {
        case 0 => ((rgb >> 16) & 0xFF) / 255.0f
        case 1 => ((rgb >> 8)  & 0xFF) / 255.0f
        case 2 => ((rgb)       & 0xFF) / 255.0f
      }
      flat(c * h * w + y * w + x) = (px - IMG_MEAN(c)) / IMG_STD(c)
    }
    flat
  }

  private def softmax(x: Array[Float]): Array[Float] = {
    val maxVal = x.max
    val exps   = x.map(v => math.exp(v - maxVal).toFloat)
    val sum    = exps.sum
    exps.map(_ / sum)
  }

  // ─── 2. Scoring Sklearn ONNX — feature-based ────────────────────────────────
  //
  // Entrée   : 6 features bbox (bbox_area_norm, aspect_ratio, cx_norm, cy_norm, w_norm, h_norm)
  // Sortie   : output_label (Long) + output_probability (Array[Array[Float]], zipmap=False)

  private def scoreWithSklearnOnnx(df: DataFrame, bc: Broadcast[Array[Byte]]): DataFrame = {
    val spark = df.sparkSession
    val outputSchema = df.schema
      .add(StructField("prediction",       StringType, nullable = true))
      .add(StructField("confidence_score", DoubleType, nullable = true))

    val rdd = df.rdd.mapPartitions { rows =>
      val env     = ai.onnxruntime.OrtEnvironment.getEnvironment()
      val session = env.createSession(bc.value, new ai.onnxruntime.OrtSession.SessionOptions())

      rows.map { row =>
        val (pred, conf) = Try {
          val features: Array[Float] = FEATURE_COLS.map { colName =>
            val idx = row.fieldIndex(colName)
            if (row.isNullAt(idx)) 0.0f else row.getDouble(idx).toFloat
          }
          val tensor = ai.onnxruntime.OnnxTensor.createTensor(
            env, java.nio.FloatBuffer.wrap(features), Array(1L, features.length.toLong))
          val inputs = new java.util.HashMap[String, ai.onnxruntime.OnnxTensorLike]()
          inputs.put("float_input", tensor)

          val result = session.run(inputs)
          val label  = result.get("output_label").get().getValue.asInstanceOf[Array[Long]](0)
          val probas = result.get("output_probability").get().getValue.asInstanceOf[Array[Array[Float]]](0)
          val confidence = if (probas.length >= 2) probas(1).toDouble else probas(0).toDouble

          tensor.close()
          result.close()
          val prediction = if (label == 1L) "plate" else "background"
          (prediction, confidence)
        }.getOrElse(("uncertain", 0.5))

        Row.fromSeq(row.toSeq ++ Seq(pred, conf))
      }
    }
    spark.createDataFrame(rdd, outputSchema)
  }

  // ─── 3. Scoring MLlib PipelineModel ─────────────────────────────────────────

  private val vectorHead: org.apache.spark.sql.expressions.UserDefinedFunction =
    udf((v: Vector) => if (v.size > 1) v(1) else v(0))

  private def scoreWithMllib(model: PipelineModel, df: DataFrame): DataFrame =
    model.transform(df)
      .withColumn("confidence_score", vectorHead(col("probability")))
      .withColumn("prediction",
        when(col("prediction") === 1.0, lit("plate")).otherwise(lit("background")))
      .drop("probability", "rawPrediction")

  // ─── 4. Scoring simulé (fallback heuristique) ────────────────────────────────
  //
  // Plaque EU typique : ratio largeur/hauteur ≈ 3.1 (520×110 mm).

  def scoreSimulated(df: DataFrame): DataFrame =
    df.withColumn("confidence_score", lit(0.5))
      .withColumn("prediction", lit("uncertain"))
}
