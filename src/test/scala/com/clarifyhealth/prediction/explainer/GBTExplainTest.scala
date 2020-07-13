package com.clarifyhealth.prediction.explainer

import java.util.UUID

import com.clarifyhealth.util.common.StageBuilder.{getExplainStages, getFeatureImportance, getPipelineStages}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.GBTClassificationModel
import org.apache.spark.ml.regression.GBTRegressionModel
import org.apache.spark.sql.QueryTest
import org.apache.spark.sql.test.SharedSparkSession

class GBTExplainTest extends QueryTest with SharedSparkSession {

  val ENSEMBLE_TYPE = "gbt"

  test("GBT regression explain") {
    spark.sharedState.cacheManager.clearCache()

    lazy val idColumn = "id"
    lazy val labelColumn = "medv"
    lazy val featuresColumn = s"features_${labelColumn}"
    lazy val prediction_column = s"prediction_${labelColumn}"

    lazy val contrib_column = s"prediction_${labelColumn}_contrib"
    lazy val contrib_column_sum = s"${contrib_column}_sum"
    lazy val contrib_column_intercept = s"${contrib_column}_intercept"

    lazy val features_importance_view = s"features_importance_${labelColumn}_view"
    lazy val predictions_view = s"prediction_${labelColumn}_view"

    val trainDf = spark.read.option("header", "true").option("inferSchema", "true")
      .csv(getClass.getResource("/regression/dataset_boston.csv").getPath)

    val featureNames = trainDf.schema.filter(x => !Array(idColumn, labelColumn).contains(x.name)).map(_.name).toArray

    val stages = getPipelineStages(Array(), featureNames, labelColumn, ENSEMBLE_TYPE, false)

    val trainPipeline = new Pipeline().setStages(stages)

    val model = trainPipeline.fit(trainDf)

    val temp_id = UUID.randomUUID()
    val gbt_model_path = s"/tmp/${temp_id}"

    val gbt_model = model.stages.last.asInstanceOf[GBTRegressionModel]
    gbt_model.write.save(gbt_model_path)

    val predictionDF = model.transform(trainDf)
    predictionDF.createOrReplaceTempView(predictions_view)

    val featureImportanceDF = getFeatureImportance(spark, gbt_model, predictionDF, featuresColumn)
    featureImportanceDF.createOrReplaceTempView(features_importance_view)

    featureImportanceDF.show(truncate = false)

    val explainStages = getExplainStages(predictions_view, features_importance_view, labelColumn,
      gbt_model_path, ENSEMBLE_TYPE, false)

    val explainPipeline = new Pipeline().setStages(explainStages)
    val explainDF = explainPipeline.fit(predictionDF).transform(predictionDF)

    val explainDFCache = explainDF.orderBy("id").limit(100).cache()

    explainDFCache.show(truncate = false)

    checkAnswer(
      explainDFCache.selectExpr(s"bround(${contrib_column_sum}+${contrib_column_intercept},5) as contribution").orderBy("id"),
      explainDFCache.selectExpr(s"bround(${prediction_column},5) as ${prediction_column}").orderBy("id")
    )

  }

  test("GBT classification explain") {
    spark.sharedState.cacheManager.clearCache()

    lazy val idColumn = "id"
    lazy val labelColumn = "outcome"
    lazy val featuresColumn = s"features_${labelColumn}"
    lazy val prediction_column = s"prediction_${labelColumn}"

    lazy val contrib_column = s"prediction_${labelColumn}_contrib"
    lazy val contrib_column_sum = s"${contrib_column}_sum"
    lazy val contrib_column_intercept = s"${contrib_column}_intercept"

    lazy val features_importance_view = s"features_importance_${labelColumn}_view"
    lazy val predictions_view = s"prediction_${labelColumn}_view"

    val trainDf = spark.read.option("header", "true").option("inferSchema", "true")
      .csv(getClass.getResource("/classification/dataset_prima_indian_diabetes.csv").getPath)

    trainDf.printSchema()

    val featureNames = trainDf.schema.filter(x => !Array(idColumn, labelColumn).contains(x.name)).map(_.name).toArray

    val stages = getPipelineStages(Array(), featureNames, labelColumn, ENSEMBLE_TYPE, true)

    val trainPipeline = new Pipeline().setStages(stages)

    val model = trainPipeline.fit(trainDf)

    val temp_id = UUID.randomUUID()
    val gbt_model_path = s"/tmp/${temp_id}"

    val gbt_model = model.stages.last.asInstanceOf[GBTClassificationModel]
    gbt_model.write.save(gbt_model_path)

    val predictionDF = model.transform(trainDf)
    predictionDF.createOrReplaceTempView(predictions_view)

    val featureImportanceDF = getFeatureImportance(spark, gbt_model, predictionDF, featuresColumn)
    featureImportanceDF.createOrReplaceTempView(features_importance_view)

    featureImportanceDF.show(truncate = false)

    val explainStages = getExplainStages(predictions_view, features_importance_view, labelColumn,
      gbt_model_path, ENSEMBLE_TYPE, true)

    val explainPipeline = new Pipeline().setStages(explainStages)
    val explainDF = explainPipeline.fit(predictionDF).transform(predictionDF)

    val explainDFCache = explainDF.orderBy("id").limit(100).cache()

    explainDFCache.show(truncate = false)

    checkAnswer(
      explainDFCache.selectExpr(s"bround(${contrib_column_sum}+${contrib_column_intercept},5) as contribution").orderBy("id"),
      explainDFCache.selectExpr(s"bround(${prediction_column},5) as ${prediction_column}").orderBy("id")
    )

  }

}

