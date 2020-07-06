package com.clarifyhealth.util.common

import com.clarifyhealth.ohe.decoder.OneHotDecoder
import com.clarifyhealth.prediction.explainer.EnsembleTreeExplainTransformer
import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassificationModel, XGBoostClassifier, XGBoostRegressionModel, XGBoostRegressor}
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, GBTClassificationModel, RandomForestClassificationModel}
import org.apache.spark.ml.{PipelineStage, PredictionModel, Transformer}
import org.apache.spark.ml.feature.{OneHotEncoderEstimator, SQLTransformer, StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types.Metadata
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.regression.{DecisionTreeRegressionModel, GBTRegressionModel, RandomForestRegressionModel}


object StageBuilder {

  def getPipelineStages(categorical_columns: Array[String], continuous_columns: Array[String], label_column: String,
                        classification: Boolean = false): Array[_ <: PipelineStage] = {

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
                       rf_model_path: String, classification: Boolean = false): Array[_ <: Transformer] = {

    val stages = Array(
      new OneHotDecoder().setOheSuffix("_OHE").setIdxSuffix("_IDX").setUnknownSuffix("Unknown"),
      new SQLTransformer().setStatement(s"CREATE OR REPLACE TEMPORARY VIEW ${predictions_view} AS SELECT * from __THIS__")
      ,
      new EnsembleTreeExplainTransformer().setFeatureImportanceView(features_importance_view)
        .setPredictionView(predictions_view).setLabel(label_column).setEnsembleType("xgboost4j")
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
}