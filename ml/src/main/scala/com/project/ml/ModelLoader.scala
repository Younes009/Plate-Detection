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
 * Charge et applique le modèle de vision (détection de plaques) au flux d'images.
 *
 * Le scoring se fait DIRECTEMENT sur les octets de l'image (`image_data`) :
 * aucun label, aucune feature bbox — le flux continu ne transporte que des
 * images brutes.
 *
 * Priorité de chargement (première correspondance gagne) :
 *   1. best_model.onnx  — CNN PyTorch exporté via ml/export_pytorch.py
 *                         Input : "image_input" [batch, 3, 256, 256] (ImageNet normalized)
 *   2. plate_model.onnx — classifieur sklearn entraîné par ml/train_model.py
 *                         sur les pixels réduits (S×S RGB, aplati CHW, /255).
 *                         Input : "float_input" [batch, 3*S*S] — S est DÉDUIT
 *                         de la forme d'entrée du modèle, pas codé en dur.
 *   3. (aucun fichier)  — fallback neutre (prediction=uncertain, confidence=0.5)
 *
 * Le fichier ONNX est broadcasté une fois depuis le driver (initialize()) ;
 * chaque partition ouvre UNE session ONNX Runtime, la ferme après consommation.
 */
object ModelLoader {

  private val PYTORCH_ONNX_FILE = "best_model.onnx"
  private val SKLEARN_ONNX_FILE = "plate_model.onnx"

  // Normalisation ImageNet (doit correspondre à export_pytorch.py)
  private val IMG_MEAN = Array(0.485f, 0.456f, 0.406f)
  private val IMG_STD  = Array(0.229f, 0.224f, 0.225f)

  @transient private var broadcastPytorchOnnx: Option[Broadcast[Array[Byte]]] = None
  @transient private var broadcastSklearnOnnx: Option[Broadcast[Array[Byte]]] = None

  def initialize(spark: SparkSession, modelPath: String = HdfsPaths.MODELS): Unit = {
    val base = modelPath.stripSuffix("/").stripSuffix("\\") + java.io.File.separator
    broadcastPytorchOnnx = tryBroadcast(spark, base + PYTORCH_ONNX_FILE)
    broadcastSklearnOnnx = tryBroadcast(spark, base + SKLEARN_ONNX_FILE)

    val mode = if (broadcastPytorchOnnx.isDefined) s"PyTorch ONNX ($PYTORCH_ONNX_FILE)"
               else if (broadcastSklearnOnnx.isDefined) s"Sklearn ONNX pixels ($SKLEARN_ONNX_FILE)"
               else "fallback neutre (aucun modèle dans MODEL/)"
    println(s"[ModelLoader] Mode actif : $mode")
  }

  def score(df: DataFrame): DataFrame = broadcastPytorchOnnx match {
    case Some(bc) => scoreWithPytorchOnnx(df, bc)
    case None => broadcastSklearnOnnx match {
      case Some(bc) => scoreWithSklearnPixels(df, bc)
      case None     => scoreFallback(df)
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

  private def withScoringSchema(df: DataFrame): StructType = df.schema
    .add(StructField("prediction",       StringType, nullable = true))
    .add(StructField("confidence_score", DoubleType, nullable = true))

  // ─── 1. Scoring PyTorch ONNX — CNN [1,3,256,256] ────────────────────────────

  private def scoreWithPytorchOnnx(df: DataFrame, bc: Broadcast[Array[Byte]]): DataFrame = {
    val spark        = df.sparkSession
    val outputSchema = withScoringSchema(df)

    val rdd = df.rdd.mapPartitions { rows =>
      val env     = ai.onnxruntime.OrtEnvironment.getEnvironment()
      val session = env.createSession(bc.value, new ai.onnxruntime.OrtSession.SessionOptions())

      val scoredRows = rows.map { row =>
        val (pred, conf) = Try {
          val imgBytes = row.getAs[Array[Byte]]("image_data")
          if (imgBytes == null || imgBytes.isEmpty) ("uncertain", 0.0)
          else {
            val img    = resize(ImageIO.read(new ByteArrayInputStream(imgBytes)), 256)
            val flat   = toFloatCHW(img, normalizeImageNet = true)
            val tensor = ai.onnxruntime.OnnxTensor.createTensor(
              env, java.nio.FloatBuffer.wrap(flat), Array(1L, 3L, 256L, 256L))
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
            val probs      = if (math.abs(rawProbs.sum - 1.0f) < 0.05f) rawProbs else softmax(rawProbs)
            val confidence = if (probs.length >= 2) probs(1).toDouble else probs(0).toDouble
            val prediction = if (confidence >= 0.5) "plate" else "background"

            tensor.close()
            result.close()
            (prediction, confidence)
          }
        }.getOrElse(("uncertain", 0.0))

        Row.fromSeq(row.toSeq ++ Seq(pred, conf))
      }
      // Ferme la session ONNX une fois la partition entièrement consommée
      scoredRows ++ { session.close(); Iterator.empty }
    }
    spark.createDataFrame(rdd, outputSchema)
  }

  // ─── 2. Scoring Sklearn ONNX — pixels réduits [1, 3*S*S] ────────────────────

  private def scoreWithSklearnPixels(df: DataFrame, bc: Broadcast[Array[Byte]]): DataFrame = {
    val spark        = df.sparkSession
    val outputSchema = withScoringSchema(df)

    val rdd = df.rdd.mapPartitions { rows =>
      val env     = ai.onnxruntime.OrtEnvironment.getEnvironment()
      val session = env.createSession(bc.value, new ai.onnxruntime.OrtSession.SessionOptions())

      // Taille d'entrée déduite du modèle : [None, 3*S*S] → S
      val inputInfo  = session.getInputInfo.values.iterator.next
      val inputShape = inputInfo.getInfo.asInstanceOf[ai.onnxruntime.TensorInfo].getShape
      val featureDim = inputShape(1).toInt
      val side       = math.sqrt(featureDim / 3.0).round.toInt
      require(3 * side * side == featureDim,
        s"[ModelLoader] Forme d'entrée inattendue : $featureDim (attendu 3*S*S)")

      val scoredRows = rows.map { row =>
        val (pred, conf) = Try {
          val imgBytes = row.getAs[Array[Byte]]("image_data")
          if (imgBytes == null || imgBytes.isEmpty) ("uncertain", 0.0)
          else {
            val img    = resize(ImageIO.read(new ByteArrayInputStream(imgBytes)), side)
            val flat   = toFloatCHW(img, normalizeImageNet = false) // /255, comme train_model.py
            val tensor = ai.onnxruntime.OnnxTensor.createTensor(
              env, java.nio.FloatBuffer.wrap(flat), Array(1L, featureDim.toLong))
            val inputs = new java.util.HashMap[String, ai.onnxruntime.OnnxTensorLike]()
            inputs.put("float_input", tensor)

            val result = session.run(inputs)
            // skl2onnx < 1.15 nomme les sorties output_label/output_probability,
            // les versions récentes label/probabilities : on accepte les deux.
            val label  = firstOutput(result, "output_label", "label")
              .getValue.asInstanceOf[Array[Long]](0)
            val probas = firstOutput(result, "output_probability", "probabilities")
              .getValue.asInstanceOf[Array[Array[Float]]](0)
            val confidence = if (probas.length >= 2) probas(1).toDouble else probas(0).toDouble

            tensor.close()
            result.close()
            (if (label == 1L) "plate" else "background", confidence)
          }
        }.getOrElse(("uncertain", 0.0))

        Row.fromSeq(row.toSeq ++ Seq(pred, conf))
      }
      scoredRows ++ { session.close(); Iterator.empty }
    }
    spark.createDataFrame(rdd, outputSchema)
  }

  private def firstOutput(result: ai.onnxruntime.OrtSession.Result,
                          names: String*): ai.onnxruntime.OnnxValue =
    names.iterator.map(result.get(_)).find(_.isPresent).map(_.get())
      .getOrElse(throw new NoSuchElementException(
        s"Sorties ONNX absentes : ${names.mkString(", ")}"))

  // ─── Prétraitement image (partagé par les deux modes de scoring) ────────────

  /** Redimensionne en S×S RGB (bilinéaire — cohérent avec PIL côté training). */
  private def resize(src: BufferedImage, side: Int): BufferedImage = {
    val out = new BufferedImage(side, side, BufferedImage.TYPE_INT_RGB)
    val g   = out.createGraphics()
    g.setRenderingHint(java.awt.RenderingHints.KEY_INTERPOLATION,
                       java.awt.RenderingHints.VALUE_INTERPOLATION_BILINEAR)
    g.drawImage(src, 0, 0, side, side, null)
    g.dispose()
    out
  }

  /** Image → tableau CHW float32, /255, normalisation ImageNet optionnelle. */
  private def toFloatCHW(img: BufferedImage, normalizeImageNet: Boolean): Array[Float] = {
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
      flat(c * h * w + y * w + x) =
        if (normalizeImageNet) (px - IMG_MEAN(c)) / IMG_STD(c) else px
    }
    flat
  }

  private def softmax(x: Array[Float]): Array[Float] = {
    val maxVal = x.max
    val exps   = x.map(v => math.exp(v - maxVal).toFloat)
    val sum    = exps.sum
    exps.map(_ / sum)
  }

  // ─── 3. Fallback neutre (aucun modèle disponible) ───────────────────────────

  def scoreFallback(df: DataFrame): DataFrame =
    df.withColumn("confidence_score", lit(0.5))
      .withColumn("prediction", lit("uncertain"))
}
