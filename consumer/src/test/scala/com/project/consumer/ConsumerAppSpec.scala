package com.project.consumer

import com.project.common.Schemas
import org.apache.spark.sql.{Row, SparkSession}
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

import java.nio.file.Files
import java.sql.Timestamp

/**
 * Tests du Consumer :
 *   1. Idempotence du sink : rejouer un même batchId ne duplique aucune ligne
 *      (garantie exactly-once de bout en bout).
 *   2. parseAndTransform produit exactement Schemas.stagingSchema.
 */
class ConsumerAppSpec extends AnyFunSuite with BeforeAndAfterAll {

  private var spark: SparkSession = _

  override def beforeAll(): Unit = {
    spark = SparkSession.builder()
      .master("local[1]")
      .appName("ConsumerAppSpec")
      .config("spark.driver.bindAddress", "127.0.0.1")
      .config("spark.driver.host", "localhost")
      .config("spark.ui.enabled", "false")
      .getOrCreate()
  }

  override def afterAll(): Unit = if (spark != null) spark.stop()

  private def stagingDf(names: String*) = {
    val now  = new Timestamp(System.currentTimeMillis())
    val rows = names.map { n =>
      Row(n, s"file:/tmp/$n.jpg", Array[Byte](1, 2, 3), 3L, now, now)
    }
    spark.createDataFrame(
      spark.sparkContext.parallelize(rows), Schemas.stagingSchema)
  }

  test("writeIdempotent : le rejeu d'un batchId déjà écrit est sauté (0 doublon)") {
    val base = Files.createTempDirectory("idempotence-test").toString
    val df   = stagingDf("img_a", "img_b", "img_c")

    assert(ConsumerApp.writeIdempotent(df, base, batchId = 7L),
      "première écriture : doit écrire")
    assert(!ConsumerApp.writeIdempotent(df, base, batchId = 7L),
      "rejeu du même batchId : doit sauter")
    assert(!ConsumerApp.writeIdempotent(df, base, batchId = 7L),
      "second rejeu : doit encore sauter")

    val written = spark.read.parquet(base)
    assert(written.count() == 3L, s"3 lignes attendues, ${written.count()} lues (doublons ?)")
    // partition par date + batch relue comme colonnes
    assert(written.columns.contains("ingest_date"))
    assert(written.columns.contains("batch_id"))
  }

  test("writeIdempotent : deux batchIds distincts coexistent") {
    val base = Files.createTempDirectory("idempotence-test2").toString
    assert(ConsumerApp.writeIdempotent(stagingDf("x1"), base, batchId = 0L))
    assert(ConsumerApp.writeIdempotent(stagingDf("x2", "x3"), base, batchId = 1L))
    assert(spark.read.parquet(base).count() == 3L)
  }

  test("parseAndTransform produit exactement Schemas.stagingSchema") {
    val now = new Timestamp(System.currentTimeMillis())
    val raw = spark.createDataFrame(
      spark.sparkContext.parallelize(Seq(
        Row("file:/tmp/car_01.jpg", now, 3L, Array[Byte](1, 2, 3)),
        Row("file:/tmp/notes.txt",  now, 2L, Array[Byte](9, 9)),   // non-image : filtré
        Row("file:/tmp/empty.png",  now, 0L, Array.empty[Byte])    // vide : filtré
      )),
      Schemas.binaryFileSchema)

    val staged = ConsumerApp.parseAndTransform(raw)

    assert(staged.schema.fieldNames.toSeq == Schemas.stagingSchema.fieldNames.toSeq)
    val rows = staged.collect()
    assert(rows.length == 1, "seule l'image non vide doit passer")
    assert(rows.head.getAs[String]("image_name") == "car_01")
  }
}
