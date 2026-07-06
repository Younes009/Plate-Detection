package com.project.common

import org.apache.spark.sql.types._

/**
 * Chemins par défaut pour un déploiement cluster HDFS.
 * En local, les applications reçoivent leurs chemins en arguments (voir main.py).
 */
object HdfsPaths {
  val SOURCE_INGRESS = "hdfs:///project/source/continuous_ingress/"
  val SOURCE_ML      = "hdfs:///project/source/ml_dataset/"
  val DEST_INGRESS   = "hdfs:///project/dest/continuous_ingress/"
  val DEST_ML        = "hdfs:///project/dest/ml_dataset/"
  val STAGING        = "hdfs:///project/staging/"
  val OUTPUT         = "hdfs:///project/output/"
  val MODELS         = "hdfs:///project/models/"
  val CHECKPOINT     = "hdfs:///project/checkpoint/"
}

object Schemas {

  /**
   * Schéma FIXE de la source Spark "binaryFile" — imposé par Spark, ne pas modifier.
   * C'est ce que le Consumer lit en streaming depuis DEST/continuous_ingress/
   * (images brutes déplacées telles quelles par le Producer via Hadoop FS).
   */
  val binaryFileSchema: StructType = StructType(Seq(
    StructField("path",             StringType,    nullable = false),
    StructField("modificationTime", TimestampType, nullable = false),
    StructField("length",           LongType,      nullable = false),
    StructField("content",          BinaryType,    nullable = true)
  ))

  /**
   * Schéma de la couche STAGING (WRITE intermédiaire du Consumer) : une ligne
   * par image propre, historique du flux d'inférence.
   *
   * PAS de colonnes label/bbox : le flux continu ne transporte QUE des images
   * brutes (les labels YOLO n'existent que dans ml_dataset, consommé par le
   * service ML Python — jamais par le stream).
   */
  val stagingSchema: StructType = StructType(Seq(
    StructField("image_name",   StringType,    nullable = true),
    StructField("image_path",   StringType,    nullable = false),
    StructField("image_data",   BinaryType,    nullable = true),
    StructField("length",       LongType,      nullable = true),
    StructField("ingested_at",  TimestampType, nullable = true),
    StructField("processed_at", TimestampType, nullable = false)
  ))

  /** Schéma OUTPUT : staging + colonnes de scoring ajoutées par ModelLoader. */
  val outputSchema: StructType = stagingSchema
    .add(StructField("prediction",       StringType, nullable = true))
    .add(StructField("confidence_score", DoubleType, nullable = true))
}
