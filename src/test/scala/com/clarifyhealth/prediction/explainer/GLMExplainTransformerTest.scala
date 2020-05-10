package com.clarifyhealth.prediction.explainer

import org.apache.spark.sql.test.SharedSparkSession
import org.apache.spark.sql.types.{
  DoubleType,
  LongType,
  StringType,
  StructField,
  StructType
}
import org.apache.spark.sql.{DataFrame, QueryTest}

import scala.collection.immutable.Nil

class GLMExplainTransformerTest extends QueryTest with SharedSparkSession {

  def initialize(): (DataFrame, DataFrame) = {
    val predictionDF = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(getClass.getResource("/basic/predictions.csv").getPath)

    predictionDF.createOrReplaceTempView("my_predictions")

    val my_schema = StructType(
      StructField("Feature_Index", LongType) ::
        StructField("Feature", StringType) ::
        StructField("Original_Feature", StringType) ::
        StructField("Coefficient", DoubleType) :: Nil
    )

    val coefficientsDF = spark.read
      .option("header", "true")
      .schema(my_schema)
      .csv(getClass.getResource("/basic/coefficients.csv").getPath)

    coefficientsDF.createOrReplaceTempView("my_coefficients")

    (predictionDF, coefficientsDF)
  }

  test("test powerHalfLink") {

    spark.sharedState.cacheManager.clearCache()
    val nested = true

    val (predictionDF, coefficientsDF) = initialize()

    val explainTransformer = new GLMExplainTransformer()
    explainTransformer.setCoefficientView("my_coefficients")
    explainTransformer.setPredictionView("my_predictions")
    // explainTransformer.setLinkFunctionType("powerHalfLink")
    explainTransformer.setFamily("tweedie")
    explainTransformer.setLinkPower(0.5)
    explainTransformer.setVariancePower(1.0)
    explainTransformer.setNested(nested)
    explainTransformer.setCalculateSum(true)
    explainTransformer.setLabel("test")

    val df = spark.emptyDataFrame
    val resultDF = explainTransformer.transform(df)

    val contribPowerHalfLink = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(getClass.getResource("/basic/contribs_power_0.5_link.csv").getPath)
      .selectExpr(
        "ccg_id",
        "bround(contrib_intercept,3) as contrib_intercept",
        "bround(contrib_sum,3) as contrib_sum",
        "bround(calculated_prediction,3) as calculated_prediction"
      )
      .orderBy("ccg_id")

    checkAnswer(
      resultDF
        .selectExpr(
          "ccg_id",
          "bround(prediction_test_contrib_intercept,3) as contrib_intercept",
          "bround(prediction_test_contrib_sum,3) as contrib_sum",
          "bround(calculated_prediction,3) as calculated_prediction"
        )
        .orderBy("ccg_id"),
      contribPowerHalfLink
    )
  }

  test("test logLink") {

    spark.sharedState.cacheManager.clearCache()
    val nested = false

    val (predictionDF, coefficientsDF) = initialize()

    val explainTransformer = new GLMExplainTransformer()
    explainTransformer.setCoefficientView("my_coefficients")
    explainTransformer.setPredictionView("my_predictions")
    explainTransformer.setLinkFunctionType("logLink")
    explainTransformer.setNested(nested)
    explainTransformer.setCalculateSum(true)
    explainTransformer.setLabel("test")

    val df = spark.emptyDataFrame
    val resultDF = explainTransformer.transform(df)

    val logLinkDF = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(getClass.getResource("/basic/contribs_log_link.csv").getPath)
      .selectExpr(
        "ccg_id",
        "bround(contrib_intercept,3) as contrib_intercept",
        "bround(contrib_sum,3) as contrib_sum",
        "bround(calculated_prediction,3) as calculated_prediction"
      )
      .orderBy("ccg_id")

    checkAnswer(
      resultDF
        .selectExpr(
          "ccg_id",
          "bround(prediction_test_contrib_intercept,3) as contrib_intercept",
          "bround(prediction_test_contrib_sum,3) as contrib_sum",
          "bround(calculated_prediction,3) as calculated_prediction"
        )
        .orderBy("ccg_id"),
      logLinkDF
    )
  }

  test("test identityLink") {

    spark.sharedState.cacheManager.clearCache()
    val nested = true

    val (predictionDF, coefficientsDF) = initialize()

    val explainTransformer = new GLMExplainTransformer()
    explainTransformer.setCoefficientView("my_coefficients")
    explainTransformer.setPredictionView("my_predictions")
    explainTransformer.setLinkFunctionType("identityLink")
    explainTransformer.setNested(nested)
    explainTransformer.setCalculateSum(true)
    explainTransformer.setLabel("test")

    val df = spark.emptyDataFrame
    val resultDF = explainTransformer.transform(df)

    val identityLinkDF = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(getClass.getResource("/basic/contribs_identity_link.csv").getPath)
      .selectExpr(
        "ccg_id",
        "bround(contrib_intercept,3) as contrib_intercept",
        "bround(contrib_sum,3) as contrib_sum",
        "bround(calculated_prediction,3) as calculated_prediction"
      )
      .orderBy("ccg_id")

    checkAnswer(
      resultDF
        .selectExpr(
          "ccg_id",
          "bround(prediction_test_contrib_intercept,3) as contrib_intercept",
          "bround(prediction_test_contrib_sum,3) as contrib_sum",
          "bround(calculated_prediction,3) as calculated_prediction"
        )
        .orderBy("ccg_id"),
      identityLinkDF
    )

  }

  test("test logitLink") {

    spark.sharedState.cacheManager.clearCache()
    val nested = false

    val (predictionDF, coefficientsDF) = initialize()

    val explainTransformer = new GLMExplainTransformer()
    explainTransformer.setCoefficientView("my_coefficients")
    explainTransformer.setPredictionView("my_predictions")
    explainTransformer.setLinkFunctionType("logitLink")
    explainTransformer.setNested(nested)
    explainTransformer.setCalculateSum(true)
    explainTransformer.setLabel("test")

    val df = spark.emptyDataFrame
    val resultDF = explainTransformer.transform(df)

    val logitLinkDF = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(getClass.getResource("/basic/contribs_logit_link.csv").getPath)
      .selectExpr(
        "ccg_id",
        "bround(contrib_intercept,3) as contrib_intercept",
        "bround(contrib_sum,3) as contrib_sum",
        "bround(calculated_prediction,3) as calculated_prediction"
      )
      .orderBy("ccg_id")

    checkAnswer(
      resultDF
        .selectExpr(
          "ccg_id",
          "bround(prediction_test_contrib_intercept,3) as contrib_intercept",
          "bround(prediction_test_contrib_sum,3) as contrib_sum",
          "bround(calculated_prediction,3) as calculated_prediction"
        )
        .orderBy("ccg_id"),
      logitLinkDF
    )

  }

  test("test inverseLink") {

    spark.sharedState.cacheManager.clearCache()
    val nested = false

    val (predictionDF, coefficientsDF) = initialize()

    val explainTransformer = new GLMExplainTransformer()
    explainTransformer.setCoefficientView("my_coefficients")
    explainTransformer.setPredictionView("my_predictions")
    explainTransformer.setLinkFunctionType("inverseLink")
    explainTransformer.setNested(nested)
    explainTransformer.setCalculateSum(true)
    explainTransformer.setLabel("test")

    val df = spark.emptyDataFrame
    val resultDF = explainTransformer.transform(df)

    val inverseLinkDF = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(getClass.getResource("/basic/contribs_inverse_link.csv").getPath)
      .selectExpr(
        "ccg_id",
        "bround(contrib_intercept,3) as contrib_intercept",
        "bround(contrib_sum,3) as contrib_sum",
        "bround(calculated_prediction,3) as calculated_prediction"
      )
      .orderBy("ccg_id")

    checkAnswer(
      resultDF
        .selectExpr(
          "ccg_id",
          "bround(prediction_test_contrib_intercept,3) as contrib_intercept",
          "bround(prediction_test_contrib_sum,3) as contrib_sum",
          "bround(calculated_prediction,3) as calculated_prediction"
        )
        .orderBy("ccg_id"),
      inverseLinkDF
    )

  }

}
