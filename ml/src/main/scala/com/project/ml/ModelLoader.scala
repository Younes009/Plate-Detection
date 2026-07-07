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
  private val LOCATOR_ONNX_FILE = "plate_locator.onnx"

  // Normalisation ImageNet (doit correspondre à export_pytorch.py)
  private val IMG_MEAN = Array(0.485f, 0.456f, 0.406f)
  private val IMG_STD  = Array(0.229f, 0.224f, 0.225f)

  @transient private var broadcastPytorchOnnx: Option[Broadcast[Array[Byte]]] = None
  @transient private var broadcastSklearnOnnx: Option[Broadcast[Array[Byte]]] = None
  @transient private var broadcastLocatorOnnx: Option[Broadcast[Array[Byte]]] = None

  def initialize(spark: SparkSession, modelPath: String = HdfsPaths.MODELS): Unit = {
    val base = modelPath.stripSuffix("/").stripSuffix("\\") + java.io.File.separator
    broadcastPytorchOnnx = tryBroadcast(spark, base + PYTORCH_ONNX_FILE)
    broadcastSklearnOnnx = tryBroadcast(spark, base + SKLEARN_ONNX_FILE)
    broadcastLocatorOnnx = tryBroadcast(spark, base + LOCATOR_ONNX_FILE)

    val mode = if (broadcastPytorchOnnx.isDefined) s"PyTorch ONNX ($PYTORCH_ONNX_FILE)"
               else if (broadcastSklearnOnnx.isDefined) s"Sklearn ONNX pixels ($SKLEARN_ONNX_FILE)"
               else "fallback neutre (aucun modèle dans MODEL/)"
    val loc = if (broadcastLocatorOnnx.isDefined) s" + localisateur ($LOCATOR_ONNX_FILE)" else ""
    println(s"[ModelLoader] Mode actif : $mode$loc")
  }

  def score(df: DataFrame): DataFrame = broadcastPytorchOnnx match {
    case Some(bc) => scoreWithPytorchOnnx(df, bc)
    case None => broadcastSklearnOnnx match {
      case Some(bc) => scoreWithSklearnPixels(df, bc, broadcastLocatorOnnx)
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

  // MÊME ordre de colonnes dans TOUS les chemins de scoring (nominal, dégradé,
  // fallback) — sinon Parquet physiquement divergents en OUTPUT (cf. audit H1).
  private def withScoringSchema(df: DataFrame): StructType = df.schema
    .add(StructField("prediction",       StringType, nullable = true))
    .add(StructField("confidence_score", DoubleType, nullable = true))
    .add(StructField("bbox_cx",          DoubleType, nullable = true))
    .add(StructField("bbox_cy",          DoubleType, nullable = true))
    .add(StructField("bbox_w",           DoubleType, nullable = true))
    .add(StructField("bbox_h",           DoubleType, nullable = true))

  private val NoBbox: Seq[Any] = Seq(null, null, null, null)

  /**
   * Ouvre une session ONNX pour la partition courante et garantit sa fermeture
   * à la fin de la tâche Spark — succès OU échec. L'ancien idiome
   * `iterator ++ { session.close(); Iterator.empty }` ne fermait la session que
   * si l'itérateur était consommé jusqu'au bout : toute exception à mi-partition
   * laissait fuir la mémoire native ONNX Runtime sur l'executor.
   * NB : OrtEnvironment est un singleton processus géré par ORT — il ne doit
   * PAS être fermé par partition.
   */
  private def newSessionClosedOnTaskEnd(env: ai.onnxruntime.OrtEnvironment,
                                        modelBytes: Array[Byte]): ai.onnxruntime.OrtSession = {
    val session = env.createSession(modelBytes, new ai.onnxruntime.OrtSession.SessionOptions())
    Option(org.apache.spark.TaskContext.get()).foreach(
      _.addTaskCompletionListener[Unit](_ => Try(session.close())))
    session
  }

  // ─── 1. Scoring PyTorch ONNX — CNN [1,3,256,256] ────────────────────────────

  private def scoreWithPytorchOnnx(df: DataFrame, bc: Broadcast[Array[Byte]]): DataFrame = {
    val spark        = df.sparkSession
    val outputSchema = withScoringSchema(df)

    val rdd = df.rdd.mapPartitions { rows =>
      val env     = ai.onnxruntime.OrtEnvironment.getEnvironment()
      val session = newSessionClosedOnTaskEnd(env, bc.value)

      rows.map { row =>
        val (pred, conf) = Try {
          val imgBytes = row.getAs[Array[Byte]]("image_data")
          if (imgBytes == null || imgBytes.isEmpty) ("uncertain", 0.0)
          else {
            val img  = resize(ImageIO.read(new ByteArrayInputStream(imgBytes)), 256)
            val flat = toFloatCHW(img, normalizeImageNet = true)
            // try/finally : les buffers natifs ORT sont libérés même si run() échoue
            var tensor: ai.onnxruntime.OnnxTensor        = null
            var result: ai.onnxruntime.OrtSession.Result = null
            try {
              tensor = ai.onnxruntime.OnnxTensor.createTensor(
                env, java.nio.FloatBuffer.wrap(flat), Array(1L, 3L, 256L, 256L))
              val inputs = new java.util.HashMap[String, ai.onnxruntime.OnnxTensorLike]()
              inputs.put("image_input", tensor)

              result  = session.run(inputs)
              val raw = result.get(0).getValue
              val rawProbs: Array[Float] = raw match {
                case a: Array[Array[Float]] => a(0)
                case a: Array[Float]        => a
                case _                      => Array(0.5f, 0.5f)
              }
              // Applique softmax si ce sont des logits (somme ≠ 1)
              val probs      = if (math.abs(rawProbs.sum - 1.0f) < 0.05f) rawProbs else softmax(rawProbs)
              val confidence = if (probs.length >= 2) probs(1).toDouble else probs(0).toDouble
              ((if (confidence >= 0.5) "plate" else "background"), confidence)
            } finally {
              if (result != null) result.close()
              if (tensor != null) tensor.close()
            }
          }
        }.getOrElse(("uncertain", 0.0))

        // Le CNN ne localise pas : bbox null (mêmes colonnes que les autres chemins)
        Row.fromSeq(row.toSeq ++ Seq(pred, conf) ++ NoBbox)
      }
    }
    spark.createDataFrame(rdd, outputSchema)
  }

  // ─── 2. Scoring Sklearn ONNX — pixels réduits [1, 3*S*S] ────────────────────

  private def scoreWithSklearnPixels(df: DataFrame, bc: Broadcast[Array[Byte]],
                                     bcLocator: Option[Broadcast[Array[Byte]]]): DataFrame = {
    val spark        = df.sparkSession
    val outputSchema = withScoringSchema(df)

    val rdd = df.rdd.mapPartitions { rows =>
      val env     = ai.onnxruntime.OrtEnvironment.getEnvironment()
      val session = newSessionClosedOnTaskEnd(env, bc.value)

      // Localisateur optionnel : seconde session. La forme d'entrée décide du
      // prétraitement — rank 4 = CNN [1,3,S,S], rank 2 = sklearn aplati [1,3*S*S].
      val locatorInfo: Option[(ai.onnxruntime.OrtSession, String, Array[Long])] =
        bcLocator.map { b =>
          val loc   = newSessionClosedOnTaskEnd(env, b.value)
          val name  = loc.getInputInfo.keySet.iterator.next
          val shape = loc.getInputInfo.values.iterator.next
            .getInfo.asInstanceOf[ai.onnxruntime.TensorInfo].getShape
          (loc, name, shape)
        }

      // Taille d'entrée du classifieur : [None, 3*S*S] → S
      val inputInfo  = session.getInputInfo.values.iterator.next
      val inputShape = inputInfo.getInfo.asInstanceOf[ai.onnxruntime.TensorInfo].getShape
      val featureDim = inputShape(1).toInt
      val side       = math.sqrt(featureDim / 3.0).round.toInt
      require(3 * side * side == featureDim,
        s"[ModelLoader] Forme d'entrée inattendue : $featureDim (attendu 3*S*S)")

      rows.map { row =>
        val (pred, conf, bbox) = Try {
          val imgBytes = row.getAs[Array[Byte]]("image_data")
          if (imgBytes == null || imgBytes.isEmpty) ("uncertain", 0.0, NoBbox)
          else {
            // Décodage UNIQUE ; chaque modèle a ensuite son propre resize
            val decoded = ImageIO.read(new ByteArrayInputStream(imgBytes))
            val flat    = toFloatCHW(resize(decoded, side), normalizeImageNet = false)
            // try/finally : les buffers natifs ORT sont libérés même si run() échoue
            var tensor:    ai.onnxruntime.OnnxTensor        = null
            var locTensor: ai.onnxruntime.OnnxTensor        = null
            var result:    ai.onnxruntime.OrtSession.Result = null
            var locRes:    ai.onnxruntime.OrtSession.Result = null
            try {
              tensor = ai.onnxruntime.OnnxTensor.createTensor(
                env, java.nio.FloatBuffer.wrap(flat), Array(1L, featureDim.toLong))
              val inputs = new java.util.HashMap[String, ai.onnxruntime.OnnxTensorLike]()
              inputs.put("float_input", tensor)

              result = session.run(inputs)
              // skl2onnx < 1.15 nomme les sorties output_label/output_probability,
              // les versions récentes label/probabilities : on accepte les deux.
              val label  = firstOutput(result, "output_label", "label")
                .getValue.asInstanceOf[Array[Long]](0)
              val probas = firstOutput(result, "output_probability", "probabilities")
                .getValue.asInstanceOf[Array[Array[Float]]](0)
              val confidence = if (probas.length >= 2) probas(1).toDouble else probas(0).toDouble
              val prediction = if (label == 1L) "plate" else "background"

              // PREDICT bbox : trace la boîte autour de la plaque détectée
              // (coordonnées YOLO normalisées, clampées [0,1])
              val bboxSeq: Seq[Any] = locatorInfo match {
                case Some((loc, inputName, shape)) if prediction == "plate" =>
                  val (locFlat, locDims) =
                    if (shape.length == 4) { // CNN [batch, 3, S, S]
                      val s = shape(2).toInt
                      (toFloatCHW(resize(decoded, s), normalizeImageNet = false),
                       Array(1L, 3L, s.toLong, s.toLong))
                    } else {                 // sklearn [batch, 3*S*S]
                      val dim = shape(1).toInt
                      val s   = math.sqrt(dim / 3.0).round.toInt
                      (toFloatCHW(resize(decoded, s), normalizeImageNet = false),
                       Array(1L, dim.toLong))
                    }
                  locTensor = ai.onnxruntime.OnnxTensor.createTensor(
                    env, java.nio.FloatBuffer.wrap(locFlat), locDims)
                  val locInputs = new java.util.HashMap[String, ai.onnxruntime.OnnxTensorLike]()
                  locInputs.put(inputName, locTensor)
                  locRes = loc.run(locInputs)
                  val coords = locRes.get(0).getValue.asInstanceOf[Array[Array[Float]]](0)
                  if (coords.length >= 4)
                    coords.take(4).toSeq.map(v => math.max(0.0, math.min(1.0, v.toDouble)): Any)
                  else NoBbox
                case _ => NoBbox
              }
              (prediction, confidence, bboxSeq)
            } finally {
              if (locRes != null) locRes.close()
              if (result != null) result.close()
              if (locTensor != null) locTensor.close()
              if (tensor != null) tensor.close()
            }
          }
        }.getOrElse(("uncertain", 0.0, NoBbox))

        Row.fromSeq(row.toSeq ++ Seq(pred, conf) ++ bbox)
      }
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

  /** Image → tableau CHW float32, /255, normalisation ImageNet optionnelle.
   *  getRGB en bloc (une seule lecture h·w) : getRGB(x,y) pixel par pixel est
   *  le goulot CPU du scoring (3·h·w appels JNI-like sur le hot path). */
  private def toFloatCHW(img: BufferedImage, normalizeImageNet: Boolean): Array[Float] = {
    val w      = img.getWidth
    val h      = img.getHeight
    val n      = h * w
    val pixels = img.getRGB(0, 0, w, h, null, 0, w)
    val flat   = new Array[Float](3 * n)
    var i = 0
    while (i < n) {
      val rgb = pixels(i)
      val r = ((rgb >> 16) & 0xFF) / 255.0f
      val g = ((rgb >> 8)  & 0xFF) / 255.0f
      val b = ( rgb        & 0xFF) / 255.0f
      if (normalizeImageNet) {
        flat(i)         = (r - IMG_MEAN(0)) / IMG_STD(0)
        flat(n + i)     = (g - IMG_MEAN(1)) / IMG_STD(1)
        flat(2 * n + i) = (b - IMG_MEAN(2)) / IMG_STD(2)
      } else {
        flat(i)         = r
        flat(n + i)     = g
        flat(2 * n + i) = b
      }
      i += 1
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

  // prediction AVANT confidence_score AVANT bbox_* : même ordre de colonnes que
  // les chemins ONNX et Schemas.outputSchema (sinon Parquet divergents en OUTPUT).
  def scoreFallback(df: DataFrame): DataFrame =
    df.withColumn("prediction", lit("uncertain"))
      .withColumn("confidence_score", lit(0.5))
      .withColumn("bbox_cx", lit(null).cast(DoubleType))
      .withColumn("bbox_cy", lit(null).cast(DoubleType))
      .withColumn("bbox_w",  lit(null).cast(DoubleType))
      .withColumn("bbox_h",  lit(null).cast(DoubleType))
}
