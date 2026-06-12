package com.project.ml

import com.project.common.HdfsPaths
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._

import scala.util.Try

/**
 * Charge un modèle d'inférence pour le scoring des plaques détectées.
 * - Si un PipelineModel MLlib est disponible sous hdfs:///project/models/plate_quality_model
 *   (ex: RandomForestClassificationModel entraîné sur les features bbox), il est utilisé.
 * - Sinon, fallback sur un scorer simulé réaliste basé sur les features bbox
 *   (aire normalisée, aspect_ratio par rapport au ratio typique d'une plaque, quality_flag).
 *
 * Ajoute toujours deux colonnes : `prediction` (string) et `confidence_score` (double).
 */
object ModelLoader {

  private val MODEL_PATH = HdfsPaths.MODELS + "plate_quality_model"

  @transient private lazy val mllibModel: Option[PipelineModel] =
    Try(PipelineModel.load(MODEL_PATH)).toOption

  private val vectorHead: UserDefinedFunction =
    udf((v: Vector) => if (v.size > 1) v(1) else v(0))

  def score(df: DataFrame): DataFrame = mllibModel match {
    case Some(model) => scoreWithMllib(model, df)
    case None        => scoreSimulated(df)
  }

  // Modèle MLlib réel (PipelineModel produisant les colonnes `prediction` et `probability`)
  private def scoreWithMllib(model: PipelineModel, df: DataFrame): DataFrame = {
    val transformed = model.transform(df)
    transformed
      .withColumn("confidence_score", vectorHead(col("probability")))
      .withColumn("prediction",
        when(col("prediction") === 1.0, lit("plate")).otherwise(lit("background")))
      .drop("probability", "rawPrediction")
  }

  // Fallback : scoring simulé réaliste à partir des features géométriques de la bbox.
  // Une plaque "typique" a un ratio largeur/hauteur proche de ~3.1 (ex: plaques EU 520x110mm).
  private def scoreSimulated(df: DataFrame): DataFrame = {
    val idealRatio = lit(3.1)
    val ratioScore = lit(1.0) - least(abs(col("aspect_ratio") - idealRatio) / idealRatio, lit(1.0))
    val areaScore  = when(col("bbox_area_norm").isNull, lit(0.0))
      .otherwise(least(col("bbox_area_norm") * lit(20.0), lit(1.0)))
    val qualityScore = when(col("quality_flag") === "OK", lit(1.0)).otherwise(lit(0.4))

    val rawScore = ratioScore * lit(0.5) + areaScore * lit(0.3) + qualityScore * lit(0.2)
    val confidence = when(col("num_boxes") > 0, greatest(least(rawScore, lit(0.99)), lit(0.05)))
      .otherwise(lit(0.0))

    df.withColumn("confidence_score", confidence)
      .withColumn("prediction",
        when(col("num_boxes") <= 0, lit("background"))
          .when(confidence >= 0.5, lit("plate"))
          .otherwise(lit("uncertain")))
  }
}
