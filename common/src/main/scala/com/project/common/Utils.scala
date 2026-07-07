package com.project.common

import org.apache.spark.sql.types._

/** Default HDFS paths for cluster deployment. Local runs pass paths as args. */
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

  /** Fixed schema of Spark's binaryFile source — do not modify. */
  val binaryFileSchema: StructType = StructType(Seq(
    StructField("path",             StringType,    nullable = false),
    StructField("modificationTime", TimestampType, nullable = false),
    StructField("length",           LongType,      nullable = false),
    StructField("content",          BinaryType,    nullable = true)
  ))

  /** STAGING layer: one row per clean image. The inference stream carries
   *  raw images only — no label/bbox columns here. */
  val stagingSchema: StructType = StructType(Seq(
    StructField("image_name",   StringType,    nullable = true),
    StructField("image_path",   StringType,    nullable = false),
    StructField("image_data",   BinaryType,    nullable = true),
    StructField("length",       LongType,      nullable = true),
    StructField("ingested_at",  TimestampType, nullable = true),
    StructField("processed_at", TimestampType, nullable = false)
  ))

  /** OUTPUT layer: staging + scoring columns. bbox_* is the PREDICTED box
   *  (normalized YOLO coords, null when no locator or below threshold). */
  val outputSchema: StructType = stagingSchema
    .add(StructField("prediction",       StringType, nullable = true))
    .add(StructField("confidence_score", DoubleType, nullable = true))
    .add(StructField("bbox_cx",          DoubleType, nullable = true))
    .add(StructField("bbox_cy",          DoubleType, nullable = true))
    .add(StructField("bbox_w",           DoubleType, nullable = true))
    .add(StructField("bbox_h",           DoubleType, nullable = true))
}
