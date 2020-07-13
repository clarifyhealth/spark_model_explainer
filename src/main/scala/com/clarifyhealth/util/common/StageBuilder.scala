package com.clarifyhealth.util.common

import com.clarifyhealth.ohe.decoder.OneHotDecoder
import com.clarifyhealth.prediction.explainer.EnsembleTreeExplainTransformer
import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassificationModel, XGBoostClassifier, XGBoostRegressionModel, XGBoostRegressor}
import org.apache.spark.ml.classification._
import org.apache.spark.ml.feature.{OneHotEncoderEstimator, SQLTransformer, StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.regression._
import org.apache.spark.ml.{PipelineStage, PredictionModel, Predictor, Transformer}
import org.apache.spark.sql.types.Metadata
import org.apache.spark.sql.{DataFrame, SparkSession}


object StageBuilder {

  def getPipelineStages(categorical_columns: Array[String], continuous_columns: Array[String], label_column: String,
                        ensembleType: String, classification: Boolean = false): Array[_ <: PipelineStage] = {

    val indexer = categorical_columns.map(c => new StringIndexer().setInputCol(c).setOutputCol(s"${c}_IDX"))

    val ohe_encoder = new OneHotEncoderEstimator()
      .setInputCols(categorical_columns.map(c => s"${c}_IDX"))
      .setOutputCols(categorical_columns.map(c => s"${c}_OHE"))
      .setDropLast(false)

    val features_column = s"features_${label_column}"
    val prediction_column = s"prediction_${label_column}"

    val feature = categorical_columns.map(c => s"${c}_OHE") ++ continuous_columns

    val assembler = new VectorAssembler().setInputCols(feature).setOutputCol(features_column)

    // NOTE: Make it work for xgboost4j : missing value will be set as NaN
    if (ensembleType == "xgboost4j")
      assembler.setHandleInvalid("keep")

    val xgb = if (classification) {
      getClassifier(ensembleType, label_column, features_column, prediction_column)
    } else {
      getPredictor(ensembleType, label_column, features_column, prediction_column)
    }
    val stages = indexer ++ Array(ohe_encoder, assembler, xgb)
    stages
  }

  def getExplainStages(predictions_view: String, features_importance_view: String, label_column: String,
                       rf_model_path: String, ensembleType: String,
                       classification: Boolean = false): Array[_ <: Transformer] = {

    val stages = Array(
      new OneHotDecoder().setOheSuffix("_OHE").setIdxSuffix("_IDX").setUnknownSuffix("Unknown"),
      new SQLTransformer().setStatement(s"CREATE OR REPLACE TEMPORARY VIEW ${predictions_view} AS SELECT * from __THIS__")
      ,
      new EnsembleTreeExplainTransformer().setFeatureImportanceView(features_importance_view)
        .setPredictionView(predictions_view).setLabel(label_column).setEnsembleType(ensembleType)
        .setModelPath(rf_model_path).setDropPathColumn(true).setIsClassification(classification)
    )
    stages
  }

  def getFeatureImportance(spark_session: SparkSession, model: PredictionModel[Vector, _], prediction_df: DataFrame,
                           features_column: String): DataFrame = {
    val featureIndexMapping: Array[Metadata] = getFeatureIndexMapping(prediction_df, features_column)
    val featureImportances = getFeatureImportances(model, featureIndexMapping.map(x => x.getString("name")))
    val data = featureIndexMapping.map { item =>
      val idx = item.getLong("idx")
      val name = item.getString("name")
      (idx, name.replaceAll("[^0-9a-zA-Z]+", "_"), name, featureImportances(idx.toInt))
    }
    spark_session.createDataFrame(data).toDF("Feature_Index", "Feature", "Original_Feature", "Importance")
  }


  def getFeatureIndexMapping(df: DataFrame, feature_column: String): Array[Metadata] = {
    val column_meta = df.schema(feature_column).metadata
    val mlAttrData = column_meta.getMetadata("ml_attr")
    val attrsData = mlAttrData.getMetadata("attrs")
    val featureIndexMapping = if (attrsData.contains("binary")) {
      attrsData.getMetadataArray("numeric") ++ attrsData.getMetadataArray("binary")
    } else {
      attrsData.getMetadataArray("numeric")
    }
    featureIndexMapping.sortBy(m => m.getLong("idx"))
  }

  def getFeatureImportances(model: PredictionModel[Vector, _], features: Array[String]): Array[Double] = {
    model match {
      case x: GBTRegressionModel => x.featureImportances.toArray
      case x: GBTClassificationModel => x.featureImportances.toArray
      case x: RandomForestRegressionModel => x.featureImportances.toArray
      case x: RandomForestClassificationModel => x.featureImportances.toArray
      case x: DecisionTreeRegressionModel => x.featureImportances.toArray
      case x: DecisionTreeClassificationModel => x.featureImportances.toArray
      case x: XGBoostRegressionModel => {
        val scores = x.nativeBooster.getScore(features, "gain")
        features.map(x => scores.getOrElse(x, 0.0))
      }
      case x: XGBoostClassificationModel => {
        val scores = x.nativeBooster.getScore(features, "gain")
        features.map(x => scores.getOrElse(x, 0.0))
      }
    }
  }

  def getClassifier(ensembleType: String, label_column: String, features_column: String, prediction_column: String): ProbabilisticClassifier[Vector, _, _] = {
    val xgbParam = Map("allow_non_zero_for_missing" -> true)

    ensembleType match {
      case "dct" => new DecisionTreeClassifier()
        .setLabelCol(label_column)
        .setFeaturesCol(features_column)
        .setPredictionCol(prediction_column)
      case "gbt" => new GBTClassifier()
        .setLabelCol(label_column)
        .setFeaturesCol(features_column)
        .setPredictionCol(prediction_column)
      case "rf" => new RandomForestClassifier()
        .setLabelCol(label_column)
        .setFeaturesCol(features_column)
        .setPredictionCol(prediction_column)
      case "xgboost4j" => new XGBoostClassifier(xgbParam)
        .setLabelCol(label_column)
        .setFeaturesCol(features_column)
        .setPredictionCol(prediction_column)
    }
  }

  def getPredictor(ensembleType: String, label_column: String, features_column: String, prediction_column: String): Predictor[Vector, _, _] = {
    val xgbParam = Map("allow_non_zero_for_missing" -> true)

    ensembleType match {
      case "dct" => new DecisionTreeRegressor()
        .setLabelCol(label_column)
        .setFeaturesCol(features_column)
        .setPredictionCol(prediction_column)
      case "gbt" => new GBTRegressor()
        .setLabelCol(label_column)
        .setFeaturesCol(features_column)
        .setPredictionCol(prediction_column)
      case "rf" => new RandomForestRegressor()
        .setLabelCol(label_column)
        .setFeaturesCol(features_column)
        .setPredictionCol(prediction_column)
      case "xgboost4j" => new XGBoostRegressor(xgbParam)
        .setLabelCol(label_column)
        .setFeaturesCol(features_column)
        .setPredictionCol(prediction_column)
    }
  }
}