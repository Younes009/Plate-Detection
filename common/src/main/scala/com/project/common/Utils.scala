package com.project.common

import org.apache.spark.sql.types._

/**
 * Chemins par défaut pour un déploiement cluster HDFS.
 * En local, les applications reçoivent leurs chemins en arguments (voir main.py).
 */
object HdfsPaths {
  val SOURCE     = "hdfs:///project/source/"
  val DEST       = "hdfs:///project/dest/"
  val STAGING    = "hdfs:///project/staging/"
  val OUTPUT     = "hdfs:///project/output/"
  val MODELS     = "hdfs:///project/models/"
  val CHECKPOINT = "hdfs:///project/checkpoint/"
}

object Schemas {

  /**
   * Schéma FIXE de la source Spark "binaryFile" — imposé par Spark, ne pas modifier.
   * C'est ce que le Consumer lit en streaming depuis DEST/ (fichiers bruts copiés
   * tels quels par le Producer via l'API Hadoop FileSystem).
   */
  val binaryFileSchema: StructType = StructType(Seq(
    StructField("path",             StringType,    nullable = false),
    StructField("modificationTime", TimestampType, nullable = false),
    StructField("length",           LongType,      nullable = false),
    StructField("content",          BinaryType,    nullable = true)
  ))

  /**
   * Schéma de la couche STAGING (WRITE 4 du Consumer) : données parsées, typées
   * et enrichies des features géométriques YOLO — une ligne par bounding box
   * (explode), les images sans label gardent une ligne avec features null.
   * C'est cette couche que le service ML Python lit (READ 5).
   */
  val stagingSchema: StructType = StructType(Seq(
    StructField("image_name",     StringType,    nullable = true),
    StructField("split",          StringType,    nullable = true),
    StructField("image_path",     StringType,    nullable = false),
    StructField("image_data",     BinaryType,    nullable = true),
    StructField("ingested_at",    TimestampType, nullable = true),
    StructField("label_path",     StringType,    nullable = true),
    StructField("label_data",     StringType,    nullable = true),
    StructField("class_id",       IntegerType,   nullable = true),
    StructField("cx_norm",        DoubleType,    nullable = true),
    StructField("cy_norm",        DoubleType,    nullable = true),
    StructField("w_norm",         DoubleType,    nullable = true),
    StructField("h_norm",         DoubleType,    nullable = true),
    StructField("bbox_area_norm", DoubleType,    nullable = true),
    StructField("aspect_ratio",   DoubleType,    nullable = true),
    StructField("processed_at",   TimestampType, nullable = false)
  ))

  /** Schéma OUTPUT : staging + colonnes de scoring ajoutées par ModelLoader. */
  val outputSchema: StructType = stagingSchema
    .add(StructField("prediction",       StringType, nullable = true))
    .add(StructField("confidence_score", DoubleType, nullable = true))
}
