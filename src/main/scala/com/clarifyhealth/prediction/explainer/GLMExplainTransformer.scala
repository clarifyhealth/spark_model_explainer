package com.clarifyhealth.prediction.explainer

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{
  DefaultParamsReadable,
  DefaultParamsWritable,
  Identifiable
}
import org.apache.spark.sql.catalyst.encoders.{ExpressionEncoder, RowEncoder}
import org.apache.spark.sql.functions.expr
import org.apache.spark.sql.types.{DataTypes, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}

import scala.collection.SortedMap

class GLMExplainTransformer(override val uid: String)
  extends Transformer
    with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("GLMExplainTransformer"))

  // Transformer Params
  // Defining a Param requires 3 elements:
  //  - Param definition
  //  - Param getter method
  //  - Param setter method
  // (The getter and setter are technically not required, but they are nice standards to follow.)

  /**
    * Param for predictionView view name.
    */
  final val predictionView: Param[String] =
    new Param[String](
      this,
      "predictionView",
      "input predictionView view name"
    )

  final def getPredictionView: String = $(predictionView)

  final def setPredictionView(value: String): GLMExplainTransformer =
    set(predictionView, value)

  /**
    * Param for coefficientView name.
    */
  final val coefficientView: Param[String] =
    new Param[String](this, "coefficientView", "input coefficient view name")

  final def getCoefficientView: String = $(coefficientView)

  final def setCoefficientView(value: String): GLMExplainTransformer =
    set(coefficientView, value)

  /**
    * Param for link function type .
    */
  final val linkFunctionType: Param[String] =
    new Param[String](this, "linkFunctionType", "input linkFunction type")

  final def getLinkFunctionType: String = $(linkFunctionType)

  final def setLinkFunctionType(value: String): GLMExplainTransformer =
    set(linkFunctionType, value)

  /**
    * Param for control the output flattens vs nested in array .
    */
  final val nested: Param[Boolean] =
    new Param[Boolean](
      this,
      "nested",
      "results nested vs flattened toggle control"
    )

  final def getNested: Boolean = $(nested)

  final def setNested(value: Boolean): GLMExplainTransformer =
    set(nested, value)

  /**
    * Param to calculate sum of contributions .
    */
  final val calculateSum: Param[Boolean] =
    new Param[Boolean](
      this,
      "calculateSum",
      "sum of all contributions to calculate toggle control"
    )

  final def getCalculateSum: Boolean = $(calculateSum)

  final def setCalculateSum(value: Boolean): GLMExplainTransformer =
    set(calculateSum, value)

  /**
    * Param for label name.
    */
  final val label: Param[String] =
    new Param[String](
      this,
      "label",
      "training label name"
    )

  final def getLabel: String = $(label)

  final def setLabel(value: String): GLMExplainTransformer =
    set(label, value)

  /**
    * Param for family name.
    */
  final val family: Param[String] =
    new Param[String](
      this,
      "family",
      "glm family name"
    )

  final def getFamily: String = $(family)

  final def setFamily(value: String): GLMExplainTransformer =
    set(family, value)

  /**
    * Param for variancePower.
    */
  final val variancePower: Param[Double] =
    new Param[Double](
      this,
      "variancePower",
      "tweedie variancePower"
    )

  final def getVariancePower: Double = $(variancePower)

  final def setVariancePower(value: Double): GLMExplainTransformer =
    set(variancePower, value)

  /**
    * Param for linkPower.
    */
  final val linkPower: Param[Double] =
    new Param[Double](
      this,
      "linkPower",
      "tweedie linkPower"
    )

  final def getLinkPower: Double = $(linkPower)

  final def setLinkPower(value: Double): GLMExplainTransformer =
    set(linkPower, value)

  // (Optional) You can set defaults for Param values if you like.
  setDefault(
    predictionView -> "predictions",
    coefficientView -> "coefficient",
    linkFunctionType -> "powerHalfLink",
    nested -> false,
    calculateSum -> false,
    label -> "test",
    family -> "gaussian",
    linkPower -> 0.0,
    variancePower -> -1.0
  )

  val epsilon: Double = 1e-16

  private val logLink: (String, String, Double) => String = {

    case ("tweedie", x, 0.0) =>
      s"""case when exp(${x}) = '-Infinity' then ${Double.MinValue}
         |when exp(${x}) = '+Infinity' then ${Double.MaxValue}
         |else exp(${x}) 
         |end""".stripMargin

    case ("gaussian", x, _) =>
      s"""case when exp(${x}) = '-Infinity' then ${Double.MinValue}
         |when exp(${x}) = '+Infinity' then ${Double.MaxValue}
         |else exp(${x}) 
         |end""".stripMargin

    case ("binomial", x, _) =>
      s"""case when exp(${x}) < ${epsilon} then ${epsilon}
         |when exp(${x}) > 1.0-${epsilon} then 1.0-${epsilon}
         |else exp(${x}) end""".stripMargin

    case ("tweedie" | "poisson" | "gamma", x, _) =>
      s"""case when exp(${x}) < ${epsilon} then ${epsilon}
         |when exp(${x}) = 'Infinity' then ${Double.MaxValue}
         |else exp(${x}) end""".stripMargin

    case (_, x, _) => s"exp(${x})"
  }

  private val logitLink: (String, String, Double) => String = {

    case ("tweedie", x, 0.0) =>
      s"""case when 1/(1+exp(-(${x}))) = '-Infinity' then ${Double.MinValue}
         |when 1/(1+exp(-(${x}))) = '+Infinity' then ${Double.MaxValue}
         |else 1/(1+exp(-(${x}))) end""".stripMargin

    case ("gaussian", x, _) =>
      s"""case when 1/(1+exp(-(${x}))) = '-Infinity' then ${Double.MinValue}
         |when 1/(1+exp(-(${x}))) = '+Infinity' then ${Double.MaxValue}
         |else 1/(1+exp(-(${x}))) end""".stripMargin

    case ("binomial", x, _) =>
      s"""case when 1/(1+exp(-(${x}))) < ${epsilon} then ${epsilon}
         |when 1/(1+exp(-(${x}))) > 1.0-${epsilon} then 1.0-${epsilon}
         |else 1/(1+exp(-(${x}))) end""".stripMargin

    case ("tweedie" | "poisson" | "gamma", x, _) =>
      s"""case when 1/(1+exp(-(${x}))) < ${epsilon} then ${epsilon}
         |when 1/(1+exp(-(${x}))) = 'Infinity' then ${Double.MaxValue}
         |else 1/(1+exp(-(${x}))) end""".stripMargin

    case (_, x, _) => s"1/(1+exp(-(${x})))"
  }

  private val identityLink: (String, String, Double) => String = {

    case ("tweedie", x, 0.0) =>
      s"""case when cast(${x} as double) = '-Infinity' then ${Double.MinValue}
         |when cast(${x} as double) = '+Infinity' then ${Double.MaxValue}
         |else cast(${x} as double) end""".stripMargin

    case ("gaussian", x, _) =>
      s"""case when cast(${x} as double) = '-Infinity' then ${Double.MinValue}
         |when cast(${x} as double) = '+Infinity' then ${Double.MaxValue}
         |else cast(${x} as double) end""".stripMargin

    case ("binomial", x, _) =>
      s"""case when cast(${x} as double) < ${epsilon} then ${epsilon}
         |when cast(${x} as double) > 1.0-${epsilon} then 1.0-${epsilon}
         |else cast(${x} as double) end""".stripMargin

    case ("tweedie" | "poisson" | "gamma", x, _) =>
      s"""case when cast(${x} as double) < ${epsilon} then ${epsilon}
         |when cast(${x} as double) = 'Infinity' then ${Double.MaxValue}
         |else cast(${x} as double) end""".stripMargin

    case (_, x, _) => s"cast(${x} as double)"
  }

  private val inverseLink: (String, String, Double) => String = {

    case ("tweedie", x, 0.0) =>
      s"""case when 1/cast(${x} as double) = '-Infinity' then ${Double.MinValue}
         |when 1/cast(${x} as double) = '+Infinity' then ${Double.MaxValue}
         |else 1/cast(${x} as double) end""".stripMargin

    case ("gaussian", x, _) =>
      s"""case when 1/cast(${x} as double)  = '-Infinity' then ${Double.MinValue}
         |when 1/cast(${x} as double)  = '+Infinity' then ${Double.MaxValue}
         |else 1/cast(${x} as double) end""".stripMargin

    case ("binomial", x, _) =>
      s"""case when 1/cast(${x} as double) < ${epsilon} then ${epsilon}
         |when 1/cast(${x} as double) > 1.0-${epsilon} then 1.0-${epsilon}
         |else 1/cast(${x} as double) end""".stripMargin

    case ("tweedie" | "poisson" | "gamma", x, _) =>
      s"""case when 1/cast(${x} as double) < ${epsilon} then ${epsilon}
         |when 1/cast(${x} as double)  = 'Infinity' then ${Double.MaxValue}
         |else 1/cast(${x} as double) end""".stripMargin

    case (_, x, _) => s"1/cast(${x} as double)"
  }

  private val otherPowerLink: (String, String, Double, Double) => String = {

    case ("tweedie", x, y, 0.0) =>
      s"""case when pow(${x},1/${y}) = '-Infinity' then ${Double.MinValue}
         |when pow(${x},1/${y}) = '+Infinity' then ${Double.MaxValue}
         |else pow(${x},1/${y}) end""".stripMargin

    case ("gaussian", x, y, _) =>
      s"""case when pow(${x},1/${y}) = '-Infinity' then ${Double.MinValue}
         |when pow(${x},1/${y}) = '+Infinity' then ${Double.MaxValue}
         |else pow(${x},1/${y}) end""".stripMargin

    case ("binomial", x, y, _) =>
      s"""case when pow(${x},1/${y}) < ${epsilon} then ${epsilon}
         |when pow(${x},1/${y}) > 1.0-${epsilon} then 1.0-${epsilon}
         |else pow(${x},1/${y}) end""".stripMargin

    case ("tweedie" | "poisson" | "gamma", x, y, _) =>
      s"""case when pow(${x},1/${y})  < ${epsilon} then ${epsilon}
         |when pow(${x},1/${y}) = 'Infinity' then ${Double.MaxValue}
         |else pow(${x},1/${y}) end""".stripMargin

    case (_, x, y, _) => s"pow(${x},1/${y})"
  }

  /**
    * Build link function expression dynamically based linkFunctionType
    *
    * @param linkFunctionType types of link function to use
    * @return
    */
  def buildLinkFunction(
                         family: String,
                         linkFunctionType: String
                       )(linkPower: Double, variancePower: Double): String => String =
    (x: String) => {
      (family, linkFunctionType, linkPower, variancePower) match {
        case ("tweedie", _, 0.0, _) => logLink(family, x, variancePower)
        case ("tweedie", _, 1.0, _) => identityLink(family, x, variancePower)
        case ("tweedie", _, 0.5, _) =>
          otherPowerLink(family, x, 0.5, variancePower)
        case ("tweedie", _, -1.0, _) => inverseLink(family, x, variancePower)
        case ("tweedie", _, y, _) => otherPowerLink(family, x, y, variancePower)
        case (_, "logLink", _, _) => logLink(family, x, -1.0)
        case (_, "logitLink", _, _) => logitLink(family, x, -1.0)
        case (_, "identityLink", _, _) => identityLink(family, x, -1.0)
        case (_, "powerHalfLink", _, _) => otherPowerLink(family, x, 0.5, -1.0)
        case (_, "inverseLink", _, _) => inverseLink(family, x, -1.0)
        case (_, "otherPowerLink", y, _) => otherPowerLink(family, x, y, -1.0)
        case _ => identityLink(family, x, -1.0)
      }
    }

  private val keepPositive: Double => Double = (temp: Double) => {
    if (temp < 0.0) 0.0 else temp
  }

  private val keepNegative: Double => Double = (temp: Double) => {
    if (temp > 0.0) 0.0 else temp
  }

  private val sigmaPositive: (Row, StructType, Boolean) => Double =
    (row: Row, schema: StructType, replaceZero: Boolean) => {
      val sigmaPos = row
        .getDouble(
          schema.fieldIndex("sigma_positive")
        )
      if (replaceZero)
        if (sigmaPos == 0.0) 1.0 else sigmaPos
      else
        sigmaPos
    }

  private val sigmaNegative: (Row, StructType, Boolean) => Double =
    (row: Row, schema: StructType, replaceZero: Boolean) => {
      val sigmaNeg = row
        .getDouble(
          schema.fieldIndex("sigma_negative")
        )
      if (replaceZero)
        if (sigmaNeg == 0.0) 1.0 else sigmaNeg
      else
        sigmaNeg
    }

  private val contribPositive: (Row, StructType) => Double =
    (row: Row, schema: StructType) => {
      val contribPos = row
        .getDouble(
          schema.fieldIndex("contrib_positive")
        )
      contribPos
    }

  private val contribNegative: (Row, StructType) => Double =
    (row: Row, schema: StructType) => {
      val contribNeg = row
        .getDouble(
          schema.fieldIndex("contrib_negative")
        )
      contribNeg
    }

  /**
    * To set flattened featureName with double data type schema
    *
    * @param df
    * @param columnNames
    * @return
    */
  private def getSchema(
                         df: DataFrame,
                         columnNames: List[String]
                       ): StructType = {
    var schema: StructType = df.schema
    columnNames.foreach {
      case (featureName) =>
        schema = schema.add(s"${featureName}", DataTypes.DoubleType, false)
    }
    schema
  }

  /**
    * To set nested array(val) schema
    *
    * @param df
    * @param columnName
    * @return
    */
  private def getSchema(df: DataFrame, columnName: String): StructType = {
    var schema: StructType = df.schema
    schema = schema.add(
      columnName,
      DataTypes.createArrayType(DataTypes.DoubleType),
      false
    )
    schema
  }

  /**
    * The encoder applies the schema based on nested vs flattened
    *
    * @param df
    * @param featureCoefficients Map(featureName->Double Value)
    * @param prefixOrColumnName  act as prefix when flattened mode else column name when nested mode
    * @param nested
    * @param addVector
    * @return
    */
  private def buildEncoder(
                            df: DataFrame,
                            featureCoefficients: SortedMap[Long, (String, Double)],
                            prefixOrColumnName: String,
                            nested: Boolean,
                            addVector: Boolean
                          ): ExpressionEncoder[Row] = {
    val schema =
      if (nested)
        getSchema(
          df,
          prefixOrColumnName
        )
      else
        getSchema(
          df,
          featureCoefficients
            .map(x => s"${prefixOrColumnName}_${x._2._1}")
            .toList
        )
    val schemaWithVector = if (addVector) {
      schema.add(
        s"${prefixOrColumnName}_vector",
        VectorType,
        false
      )
    } else {
      schema
    }
    RowEncoder.apply(schemaWithVector)
  }

  // Transformer requires 3 methods:
  //  - transform
  //  - transformSchema
  //  - copy

  /**
    * This method implements the main transformation.
    * Its required semantics are fully defined by the method API: take a Dataset or DataFrame,
    * and return a DataFrame.
    *
    * Most Transformers are 1-to-1 row mappings which add one or more new columns and do not
    * remove any columns.  However, this restriction is not required.  This example does a flatMap,
    * so we could either (a) drop other columns or (b) keep other columns, making copies of values
    * in each row as it expands to multiple rows in the flatMap.  We do (a) for simplicity.
    */
  override def transform(dataset: Dataset[_]): DataFrame = {

    val linkFunction =
      buildLinkFunction($(family), $(linkFunctionType))(
        $(linkPower),
        $(variancePower)
      )

    val coefficients = dataset.sqlContext
      .table($(coefficientView))
      .select("Feature_Index", "Feature", "Coefficient")
      .orderBy("Feature_Index")
      .collect()

    val allCoefficients = SortedMap(
      coefficients
        .map(
          row =>
            row.getAs[Long](0) -> (row.getAs[String](1), row.getAs[Double](2))
        ): _*
    )

    val intercept = allCoefficients.getOrElse(-1, ("Intercept", 0))._2

    val featureCoefficients =
      allCoefficients.filter(x => x._2._1 != "Intercept")

    val predictions = dataset.sqlContext.table($(predictionView))

    val df = calculateLinearContributions(
      predictions,
      featureCoefficients,
      "linear_contrib",
      $(nested)
    )

    val dfWithSigma =
      calculateSigma(df, featureCoefficients, "linear_contrib", $(nested))

    val linearSumDf = dfWithSigma.withColumn(
      "linear_sum",
      expr(s"sigma+$intercept")
    )

    val predDf = linearSumDf.withColumn(
      "calculated_prediction",
      expr(linkFunction(s"sigma+$intercept"))
    )

    val predPosDF = predDf.withColumn(
      "prediction_positive",
      expr(linkFunction(s"sigma_positive+$intercept"))
    )

    val predNegDF = predPosDF.withColumn(
      "prediction_negative",
      expr(linkFunction(s"sigma_negative+$intercept"))
    )

    val contribInterceptDF = predNegDF.withColumn(
      "contrib_intercept",
      expr(linkFunction(s"$intercept"))
    )

    val deficitDF = contribInterceptDF.withColumn(
      "deficit",
      expr(
        "calculated_prediction + contrib_intercept - (prediction_positive + prediction_negative)"
      )
    )

    val contribPosDF = deficitDF.withColumn(
      "contrib_positive",
      expr("prediction_positive - contrib_intercept + deficit / 2")
    )
    val contribNegsDF = contribPosDF.withColumn(
      "contrib_negative",
      expr("prediction_negative - contrib_intercept + deficit / 2")
    )

    /*
     calculate contribution of each feature in a row
     */
    val contributionsDF =
      calculateContributions(
        contribNegsDF,
        featureCoefficients,
        "linear_contrib",
        $(nested)
      )

    val finalDF = if ($(calculateSum)) {
      /*
        calculate sum(contribution of each feature) in a row
       */
      val contributionTotalDF = calculateTotalContrib(
        contributionsDF,
        featureCoefficients,
        "contrib",
        $(nested)
      )
      contributionTotalDF
    } else {
      contributionsDF
    }

    val finalColRenamedDF =
      finalDF.transform(appendLabelToColumnNames($(label)))

    finalColRenamedDF.createOrReplaceTempView($(predictionView))

    finalColRenamedDF

  }

  /**
    * The method to prefix column with label
    *
    * @param label
    * @param df
    * @return
    */
  def appendLabelToColumnNames(label: String)(df: DataFrame): DataFrame = {
    val contribColumns =
      List("sigma", "linear_sum", "contrib", "contrib_intercept", "contrib_sum")
    val filteredColumns = df.columns.filter(x => contribColumns.contains(x))
    filteredColumns.foldLeft(df) { (memoDF, colName) =>
      memoDF.withColumnRenamed(colName, s"prediction_${label}_${colName}")
    }
  }

  /**
    * This is the main entry point to calculate linear contribution of each feature
    *
    * @param df
    * @param featureCoefficients
    * @param prefixOrColumnName
    * @param nested
    * @return
    */
  private def calculateLinearContributions(
                                            df: DataFrame,
                                            featureCoefficients: SortedMap[Long, (String, Double)],
                                            prefixOrColumnName: String,
                                            nested: Boolean
                                          ): DataFrame = {
    val encoder =
      buildEncoder(df, featureCoefficients, prefixOrColumnName, nested, false)
    val func =
      mappingLinearContributionsRows(df.schema, nested)(featureCoefficients)
    df.mapPartitions(x => x.map(func))(encoder)
  }

  /*
    Map over Rows and features to calculate linear contribution of each feature flattened and nested mode
    ----------------------------------------------------------------------
   */
  private val mappingLinearContributionsRows: (
    StructType,
      Boolean
    ) => SortedMap[Long, (String, Double)] => Row => Row =
    (schema, nested) =>
      (featureCoefficients) =>
        (row) => {
          val calculate: List[Double] = featureCoefficients.map {
            case (_, (featureName, coefficient)) =>
              row
                .get(schema.fieldIndex(featureName))
                .toString
                .toDouble * coefficient
          }.toList
          if (nested) {
            Row.merge(row, Row(calculate))
          } else {
            Row.merge(row, Row.fromSeq(calculate))
          }
        }

  /**
    * This is the main entry point to calculate sigma, sigma+ve, sigma-ve
    *
    * @param df
    * @param featureCoefficients
    * @param prefixOrColumnName
    * @param nested
    * @return
    */
  private def calculateSigma(
                              df: DataFrame,
                              featureCoefficients: SortedMap[Long, (String, Double)],
                              prefixOrColumnName: String,
                              nested: Boolean
                            ): DataFrame = {
    val encoder =
      RowEncoder.apply(
        getSchema(df, List("sigma", "sigma_positive", "sigma_negative"))
      )
    if (nested) {
      val func = mappingNestedSigmaRows(df.schema)(prefixOrColumnName)
      df.mapPartitions(x => x.map(func))(encoder)
    } else {
      val func =
        mappingSigmaRows(df.schema)(prefixOrColumnName, featureCoefficients)
      df.mapPartitions(x => x.map(func))(encoder)
    }
  }

  /*
    Map over Rows and features to calculate sigma, sigma+ve, sigma-ve in flattened mode
    ----------------------------------------------------------------------
   */
  private val mappingSigmaRows: StructType => (
    String,
      SortedMap[Long, (String, Double)]
    ) => Row => Row =
    (schema) =>
      (prefixOrColumnName, featureCoefficients) =>
        (row) => {
          val calculate: List[Double] = List(
            featureCoefficients.map {
              case (_, (featureName, _)) =>
                row
                  .getDouble(
                    schema.fieldIndex(s"${prefixOrColumnName}_${featureName}")
                  )
            }.sum,
            featureCoefficients.map {
              case (_, (featureName, _)) =>
                val temp =
                  row.getDouble(
                    schema.fieldIndex(s"${prefixOrColumnName}_${featureName}")
                  )
                keepPositive(temp)
            }.sum,
            featureCoefficients.map {
              case (_, (featureName, _)) =>
                val temp =
                  row.getDouble(
                    schema.fieldIndex(s"${prefixOrColumnName}_${featureName}")
                  )
                keepNegative(temp)
            }.sum
          )
          Row.merge(row, Row.fromSeq(calculate))
        }

  /*
    Map over Rows and features to calculate sigma, sigma+ve, sigma-ve in nested mode
    ----------------------------------------------------------------------
   */
  private val mappingNestedSigmaRows: StructType => String => Row => Row =
    (schema) =>
      (prefixOrColumnName) =>
        (row) => {
          // retrieve the linear contributions from row(feature_index)
          val linearContributions =
            row.getSeq[Double](schema.fieldIndex(prefixOrColumnName))
          val calculate: List[Double] = List(
            // sum of all linear contributions
            linearContributions.sum,
            // sum of all positive linear contributions
            linearContributions.map {
              case (linearContrib) =>
                keepPositive(linearContrib)
            }.sum,
            // sum of all negative linear contributions
            linearContributions.map {
              case (linearContrib) =>
                keepNegative(linearContrib)
            }.sum
          )
          Row.merge(row, Row.fromSeq(calculate))
        }

  /**
    * This is the main entry point to calculate final contribution of each feature
    *
    * @param df
    * @param featureCoefficients
    * @param prefixOrColumnName
    * @param nested
    * @return
    */
  private def calculateContributions(
                                      df: DataFrame,
                                      featureCoefficients: SortedMap[Long, (String, Double)],
                                      prefixOrColumnName: String,
                                      nested: Boolean
                                    ): DataFrame = {
    val encoder =
      buildEncoder(df, featureCoefficients, "contrib", nested, true)
    if (nested) {
      val func = mappingContributionsNestedRows(df.schema)(prefixOrColumnName)
      df.mapPartitions(x => x.map(func))(encoder)
    } else {
      val func = mappingContributionsRows(df.schema)(
        prefixOrColumnName,
        featureCoefficients
      )
      df.mapPartitions(x => x.map(func))(encoder)
    }
  }

  /*
    Map over Rows and features to calculate final contribution of each feature flattened mode
    ----------------------------------------------------------------------
   */
  private val mappingContributionsRows: StructType => (
    String,
      SortedMap[Long, (String, Double)]
    ) => Row => Row =
    (schema) =>
      (prefixOrColumnName, featureCoefficients) =>
        (row) => {
          val calculate: List[Double] = featureCoefficients.map {
            case (_, (featureName, _)) =>
              // retrieve the linear contributions from prefixOrColumnName_featureName column flattened mode
              val linearContribution = row.getDouble(
                schema.fieldIndex(s"${prefixOrColumnName}_${featureName}")
              )
              // contribution double for flattened mode
              calculateContributionsInternal(
                linearContribution,
                row,
                schema
              )
          }.toList
          Row.merge(
            row,
            Row.fromSeq(calculate :+ Vectors.dense(calculate.toArray))
          )
        }
  /*
    Map over Rows and features to calculate final contribution of each feature nested mode
    ----------------------------------------------------------------------
   */
  private val mappingContributionsNestedRows
  : StructType => String => Row => Row =
    (schema) =>
      (prefixOrColumnName) =>
        (row) => {
          // retrieve the linear contributions from Seq(value)
          val linearContributions =
            row.getSeq[Double](schema.fieldIndex(prefixOrColumnName))

          val calculate: Seq[Double] = linearContributions.map {
            // contribution double for nested mode
            case (linearContribution) =>
              calculateContributionsInternal(
                linearContribution,
                row,
                schema
              )
          }
          Row.merge(
            row,
            Row.fromSeq(List(calculate, Vectors.dense(calculate.toArray)))
          )
        }

  private val calculateContributionsInternal
  : (Double, Row, StructType) => Double =
    (
      linearContribution: Double,
      row: Row,
      schema: StructType
    ) => {
      val sigmaPosZeroReplace = sigmaPositive(row, schema, true)
      val sigmaNegZeroReplace = sigmaNegative(row, schema, true)

      val contribPos = contribPositive(row, schema)
      val contribNeg = contribNegative(row, schema)

      (keepPositive(linearContribution) * contribPos / sigmaPosZeroReplace +
        keepNegative(linearContribution) * contribNeg / sigmaNegZeroReplace)
    }

  def calculateTotalContrib(
                             df: DataFrame,
                             featureCoefficients: SortedMap[Long, (String, Double)],
                             prefixOrColumnName: String,
                             nested: Boolean
                           ): DataFrame = {
    val encoder =
      RowEncoder.apply(getSchema(df, List("contrib_sum")))
    val func =
      mappingSumRows(df.schema, nested)(prefixOrColumnName, featureCoefficients)
    df.mapPartitions(x => x.map(func))(encoder)
  }

  private val mappingSumRows: (StructType, Boolean) => (
    String,
      SortedMap[Long, (String, Double)]
    ) => Row => Row =
    (schema, nested) =>
      (prefixOrColumnName, featureCoefficients) =>
        (row) => {
          val calculate =
            if (nested)
              row.getSeq[Double](schema.fieldIndex(prefixOrColumnName)).sum
            else
              featureCoefficients.map {
                case (_, (featureName, _)) =>
                  row
                    .getDouble(
                      schema.fieldIndex(s"${prefixOrColumnName}_${featureName}")
                    )
              }.sum
          Row.merge(row, Row(calculate))
        }

  /**
    * Check transform validity and derive the output schema from the input schema.
    *
    * We check validity for interactions between parameters during `transformSchema` and
    * raise an exception if any parameter value is invalid. Parameter value checks which
    * do not depend on other parameters are handled by `Param.validate()`.
    *
    * Typical implementation should first conduct verification on schema change and parameter
    * validity, including complex parameter interaction checks.
    */
  override def transformSchema(schema: StructType): StructType = {
    schema
  }

  /**
    * Creates a copy of this instance.
    * Requirements:
    *  - The copy must have the same UID.
    *  - The copy must have the same Params, with some possibly overwritten by the `extra`
    * argument.
    *  - This should do a deep copy of any data members which are mutable.  That said,
    * Transformers should generally be immutable (except for Params), so the `defaultCopy`
    * method often suffices.
    *
    * @param extra Param values which will overwrite Params in the copy.
    */
  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)
}

object GLMExplainTransformer
  extends DefaultParamsReadable[GLMExplainTransformer] {
  override def load(path: String): GLMExplainTransformer = super.load(path)
}
