package com.clarifyhealth.prediction.explainer

import org.apache.spark.sql.QueryTest
import org.apache.spark.sql.test.SharedSparkSession
import org.apache.spark.sql.types.{DoubleType, IntegerType, StructField, StructType}
import com.clarifyhealth.util.common.StageBuilder.getPipelineStages
import org.apache.spark.ml.Pipeline

class XGBoostExplainTest extends QueryTest with SharedSparkSession {

  test("xgboost4j regression explain") {
    spark.sharedState.cacheManager.clearCache()

    val labelName = "fare_amount"

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
        StructField(labelName, DoubleType),
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
    
    val featureNames = trainDf.schema.filter(_.name != labelName).map(_.name).toArray

    val stages = getPipelineStages(Array(), featureNames, labelName, false)

    val pipeline = new Pipeline()
    pipeline.setStages(stages)

    val model = pipeline.fit(trainDf)

    val df = model.transform(trainDf)

    df.show()

  }

}

