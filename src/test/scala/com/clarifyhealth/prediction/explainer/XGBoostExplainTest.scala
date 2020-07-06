package com.clarifyhealth.prediction.explainer

import org.apache.spark.sql.QueryTest
import org.apache.spark.sql.test.SharedSparkSession
import org.apache.spark.sql.types.{DoubleType, IntegerType, StructField, StructType}
import com.clarifyhealth.util.common.StageBuilder.{getExplainStages, getFeatureImportance, getPipelineStages}
import ml.dmlc.xgboost4j.scala.spark.XGBoostRegressionModel
import org.apache.spark.ml.Pipeline
import java.util.UUID

class XGBoostExplainTest extends QueryTest with SharedSparkSession {

  ignore("xgboost4j regression explain") {
    spark.sharedState.cacheManager.clearCache()

    lazy val labelColumn = "fare_amount"
    lazy val featuresColumn = s"features_${labelColumn}"
    lazy val features_importance_view = s"features_importance_${labelColumn}_view"
    lazy val predictions_view = s"prediction_${labelColumn}_view"
    lazy val contrib_column = s"prediction_${labelColumn}_contrib"
    lazy val prediction_column = s"prediction_${labelColumn}"

    lazy val schema =
      StructType(Array(
        StructField("vendor_id", DoubleType),
        StructField("passenger_count", DoubleType),
        StructField("trip_distance", DoubleType),
        StructField("pickup_longitude", DoubleType),
        StructField("pickup_latitude", DoubleType),
        StructField("rate_code", DoubleType),
        StructField("store_and_fwd", DoubleType),
        StructField("dropoff_longitude", DoubleType),
        StructField("dropoff_latitude", DoubleType),
        StructField(labelColumn, DoubleType),
        StructField("hour", DoubleType),
        StructField("year", IntegerType),
        StructField("month", IntegerType),
        StructField("day", DoubleType),
        StructField("day_of_week", DoubleType),
        StructField("is_weekend", DoubleType)
      ))

    val trainDf = spark.read
      .schema(schema)
      .csv(getClass.getResource("/basic/taxi_small.csv").getPath)

    val featureNames = trainDf.schema.filter(_.name != labelColumn).map(_.name).toArray

    val stages = getPipelineStages(Array(), featureNames, labelColumn, false)

    val trainPipeline = new Pipeline().setStages(stages)

    val model = trainPipeline.fit(trainDf)

    val temp_id = UUID.randomUUID()
    val xgb_model_path = s"/tmp/${temp_id}"

    val xgb_model = model.stages.last.asInstanceOf[XGBoostRegressionModel]
    xgb_model.write.save(xgb_model_path)

    val predictionDF = model.transform(trainDf)
    predictionDF.createOrReplaceTempView(predictions_view)

    val featureImportanceDF = getFeatureImportance(spark, xgb_model, predictionDF, featuresColumn)
    featureImportanceDF.createOrReplaceTempView(features_importance_view)

    featureImportanceDF.show(truncate = false)

    val explainStages = getExplainStages(predictions_view, features_importance_view, labelColumn, xgb_model_path)

    val explainPipeline = new Pipeline().setStages(explainStages)
    val explainDF = explainPipeline.fit(predictionDF).transform(predictionDF)

    explainDF.show()

  }

}

