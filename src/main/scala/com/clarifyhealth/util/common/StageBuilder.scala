package com.clarifyhealth.util.common

import com.clarifyhealth.ohe.decoder.OneHotDecoder
import com.clarifyhealth.prediction.explainer.EnsembleTreeExplainTransformer
import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassifier, XGBoostRegressor}
import org.apache.spark.ml.{PipelineStage, Transformer}
import org.apache.spark.ml.feature.{OneHotEncoderEstimator, SQLTransformer, StringIndexer, VectorAssembler}
import org.apache.spark.ml.util.DefaultParamsWritable

object StageBuilder {

  def getPipelineStages(categorical_columns: Array[String], continuous_columns: Array[String], label_column: String,
                        classification: Boolean = false): Array[PipelineStage with DefaultParamsWritable] = {

    val indexer = categorical_columns.map(c => new StringIndexer().setInputCol(c).setOutputCol(s"${c}_IDX"))

    val ohe_encoder = new OneHotEncoderEstimator()
      .setInputCols(categorical_columns.map(c => s"${c}_IDX"))
      .setOutputCols(categorical_columns.map(c => s"${c}_OHE"))
      .setDropLast(false)

    val features_column = s"features_${label_column}"
    val prediction_column = s"prediction_${label_column}"

    val feature = categorical_columns.map(c => s"${c}_OHE") ++ continuous_columns

    val assembler = new VectorAssembler().setInputCols(feature).setOutputCol(features_column)

    val xgb = if (classification) {
      new XGBoostClassifier().setLabelCol(label_column)
        .setFeaturesCol(features_column)
        .setPredictionCol(prediction_column)
    } else {
      new XGBoostRegressor().setLabelCol(label_column)
        .setFeaturesCol(features_column)
        .setPredictionCol(prediction_column)
    }
    val stages = indexer ++ Array(ohe_encoder, assembler, xgb)
    stages
  }

  def getExplainStages(predictions_view: String, features_importance_view: String, label_column: String,
                       rf_model_path: String, classification: Boolean = false): Array[Transformer with DefaultParamsWritable] = {

    val stages = Array(
      new OneHotDecoder().setOheSuffix("_OHE").setIdxSuffix("_IDX").setUnknownSuffix("UUnknownn"),
      new SQLTransformer().setStatement(s"CREATE OR REPLACE TEMPORARY VIEW ${predictions_view} AS SELECT * from __THIS__")
      ,
      new EnsembleTreeExplainTransformer().setFeatureImportanceView(features_importance_view)
        .setPredictionView(predictions_view).setLabel(label_column).setEnsembleType("xgboost4j")
        .setModelPath(rf_model_path).setDropPathColumn(true).setIsClassification(classification)
    )
    stages
  }
}
