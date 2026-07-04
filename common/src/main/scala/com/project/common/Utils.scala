package com.project.common

import org.apache.spark.sql.types._

object HdfsPaths {
  val SOURCE     = "hdfs:///project/source/"
  val DEST       = "hdfs:///project/dest/"
  val OUTPUT     = "hdfs:///project/output/"
  val MODELS     = "hdfs:///project/models/"
  val CHECKPOINT = "hdfs:///project/checkpoint/"
}

object Schemas {

  // Schema écrit par le Producer : images brutes, aucune transformation
  val destSchema: StructType = StructType(Seq(
    StructField("image_name",  StringType,    nullable = false),
    StructField("split",       StringType,    nullable = false),
    StructField("image_path",  StringType,    nullable = false),
    StructField("image_data",  BinaryType,    nullable = false),
    StructField("ingested_at", TimestampType, nullable = false)
  ))

  // Schema écrit par le Consumer : destSchema + jointure label + scoring
  val outputSchema: StructType = destSchema
    .add(StructField("label_path",       StringType,    nullable = true))
    .add(StructField("label_data",       StringType,    nullable = true))
    .add(StructField("prediction",       StringType,    nullable = true))
    .add(StructField("confidence_score", DoubleType,    nullable = true))
    .add(StructField("processed_at",     TimestampType, nullable = false))
}
