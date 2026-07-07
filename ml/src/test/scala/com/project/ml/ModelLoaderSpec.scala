package com.project.ml

import com.project.common.Schemas
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite

import javax.imageio.ImageIO
import scala.io.Source

/**
 * Tests du contrat de scoring :
 *   1. Parité de prétraitement Python (PIL) ↔ Scala (AWT) — la fixture
 *      parity/ est générée par le pipeline EXACT de train_model.image_to_features.
 *   2. scoreFallback produit exactement Schemas.outputSchema (ordre + types) —
 *      un fallback divergent corrompt physiquement les Parquet d'OUTPUT.
 */
class ModelLoaderSpec extends AnyFunSuite {

  test("prétraitement Scala ≈ prétraitement Python (resize bilinéaire + CHW /255)") {
    val img = ImageIO.read(getClass.getResourceAsStream("/parity/gradient.png"))
    val actual = ModelLoader.toFloatCHW(ModelLoader.resize(img, 32))

    val src = Source.fromInputStream(
      getClass.getResourceAsStream("/parity/expected_features.txt"), "UTF-8")
    val expected = try src.getLines().map(_.toFloat).toArray finally src.close()

    assert(actual.length == expected.length,
      s"dimensions divergentes : ${actual.length} vs ${expected.length}")
    // Les implémentations bilinéaires PIL et AWT diffèrent légèrement :
    // on borne la dérive moyenne (une vraie désynchronisation — BICUBIC vs
    // BILINEAR, HWC vs CHW, oubli du /255 — explose largement ce seuil).
    val meanAbsDiff = actual.zip(expected).map { case (a, e) => math.abs(a - e) }.sum / actual.length
    assert(meanAbsDiff < 0.03, f"dérive moyenne trop élevée : $meanAbsDiff%.4f (seuil 0.03)")

    val maxDiff = actual.zip(expected).map { case (a, e) => math.abs(a - e) }.max
    assert(maxDiff < 0.25, f"dérive max trop élevée : $maxDiff%.4f (seuil 0.25)")
  }

  test("bestDetection décode [1,5,N] YOLO (pixels + argmax score)") {
    def approx(actual: Seq[Any], expected: Seq[Double]): Unit =
      actual.map(_.asInstanceOf[Double]).zip(expected).foreach { case (a, e) =>
        assert(math.abs(a - e) < 1e-6, s"$a ≠ $e")
      }

    val yolo: Array[Array[Array[Float]]] = Array(Array(
      Array(64f, 128f),   // cx
      Array(64f, 128f),   // cy
      Array(32f, 64f),    // w
      Array(16f, 32f),    // h
      Array(0.2f, 0.9f))) // scores → column 1 wins
    val (bbox, score) = ModelLoader.bestDetection(yolo, 256)
    approx(bbox, Seq(0.5, 0.5, 0.25, 0.125))
    assert(math.abs(score - 0.9) < 1e-6)

    // unknown format -> no bbox, zero confidence
    val (none, zero) = ModelLoader.bestDetection(Array(1f, 2f), 256)
    assert(none.forall(_ == null) && zero == 0.0)
  }

  test("scoreFallback produit exactement Schemas.outputSchema") {
    val spark = SparkSession.builder()
      .master("local[1]")
      .appName("ModelLoaderSpec")
      .config("spark.driver.bindAddress", "127.0.0.1")
      .config("spark.driver.host", "localhost")
      .config("spark.ui.enabled", "false")
      .getOrCreate()
    try {
      val empty  = spark.createDataFrame(
        spark.sparkContext.emptyRDD[org.apache.spark.sql.Row], Schemas.stagingSchema)
      val scored = ModelLoader.scoreFallback(empty)

      assert(scored.schema.fieldNames.toSeq == Schemas.outputSchema.fieldNames.toSeq,
        "ordre/noms de colonnes divergents entre scoreFallback et outputSchema")
      val actualTypes   = scored.schema.fields.map(f => (f.name, f.dataType)).toSeq
      val expectedTypes = Schemas.outputSchema.fields.map(f => (f.name, f.dataType)).toSeq
      assert(actualTypes == expectedTypes,
        "types de colonnes divergents entre scoreFallback et outputSchema")
    } finally spark.stop()
  }
}
