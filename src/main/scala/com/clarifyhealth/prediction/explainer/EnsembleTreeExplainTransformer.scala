package com.clarifyhealth.prediction.explainer

import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassificationModel, XGBoostRegressionModel}
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, GBTClassificationModel, RandomForestClassificationModel}
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.{SQLDataTypes, Vector, Vectors}
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.regression.{DecisionTreeRegressionModel, GBTRegressionModel, RandomForestRegressionModel}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.ml.{PredictionModel, Transformer}
import org.apache.spark.sql.catalyst.encoders.{ExpressionEncoder, RowEncoder}
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Row}

import scala.collection.SortedMap
import scala.collection.immutable.Nil

class EnsembleTreeExplainTransformer(override val uid: String)
  extends Transformer
    with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("EnsembleTreeExplainTransformer"))

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

  final def setPredictionView(value: String): EnsembleTreeExplainTransformer =
    set(predictionView, value)

  /**
    * Param for featureImportance name.
    */
  final val featureImportanceView: Param[String] =
    new Param[String](this, "featureImportanceView", "input featureImportance view name")

  final def getFeatureImportanceView: String = $(featureImportanceView)

  final def setFeatureImportanceView(value: String): EnsembleTreeExplainTransformer =
    set(featureImportanceView, value)

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

  final def setLabel(value: String): EnsembleTreeExplainTransformer =
    set(label, value)

  /**
    * Param for modelPath.
    */
  final val modelPath: Param[String] =
    new Param[String](
      this,
      "modelPath",
      "fitted model path"
    )

  final def getModelPath: String = $(modelPath)

  final def setModelPath(value: String): EnsembleTreeExplainTransformer =
    set(modelPath, value)

  /**
    * Param for control to drop paths column
    */
  final val dropPathColumn: Param[Boolean] =
    new Param[Boolean](
      this,
      "dropPathColumn",
      "control to drop path column"
    )

  final def getDropPathColumn: Boolean = $(dropPathColumn)

  final def setDropPathColumn(value: Boolean): EnsembleTreeExplainTransformer =
    set(dropPathColumn, value)

  /**
    * Param for control to model classification vs regression
    */
  final val isClassification: Param[Boolean] =
    new Param[Boolean](
      this,
      "isClassification",
      "is classification  type or else regression "
    )

  final def getIsClassification: Boolean = $(isClassification)

  final def setIsClassification(
                                 value: Boolean
                               ): EnsembleTreeExplainTransformer =
    set(isClassification, value)

  /**
    * Param for ensembleType
    */
  final val ensembleType: Param[String] =
    new Param[String](
      this,
      "ensembleType",
      "input ensembleType RF or GDB etc"
    )

  final def getEnsembleType: String = $(ensembleType)

  final def setEnsembleType(value: String): EnsembleTreeExplainTransformer =
    set(ensembleType, value)


  // (Optional) You can set defaults for Param values if you like.
  setDefault(
    predictionView -> "predictions",
    featureImportanceView -> "featureImportance",
    label -> "test",
    modelPath -> "modelPath",
    dropPathColumn -> true,
    isClassification -> false,
    ensembleType -> "rf"
  )

  /**
    * To set  map(key,struct) schema for paths column
    *
    * @param df
    * @param columnName
    * @return
    */
  private def getPathsSchema(df: DataFrame, columnName: String): StructType = {
    var schema: StructType = df.schema
    schema = schema.add(
      columnName,
      DataTypes.createMapType(
        LongType,
        StructType(
          StructField("inclusion_index", LongType) ::
            StructField("inclusion_path", SQLDataTypes.VectorType) ::
            StructField("exclusion_path", SQLDataTypes.VectorType) :: Nil
        )
      ),
      false
    )
    schema
  }

  /**
    * The encoder applies the schema to paths column
    *
    * @param df
    * @param columnName column name / label as prefix
    * @return
    */
  private def buildPathsEncoder(
                                 df: DataFrame,
                                 columnName: String
                               ): ExpressionEncoder[Row] = {
    val newSchema = getPathsSchema(df, columnName)
    RowEncoder.apply(newSchema)
  }

  /**
    * To set nested array(val) schema
    *
    * @param df
    * @param columnName column name / label as prefix
    * @return
    */
  private def getContribSchema(
                                df: DataFrame,
                                columnName: String
                              ): StructType = {
    var schema: StructType = df.schema
    schema = schema.add(
      columnName,
      DataTypes.createArrayType(DoubleType),
      false
    )
    schema = schema.add(s"${columnName}_sum", DoubleType, false)
    schema = schema.add(s"${columnName}_vector", VectorType, false)
    schema
  }

  /**
    * The encoder applies the schema based on nested vs flattened
    *
    * @param df
    * @param columnName act as prefix when flattened mode else column name when nested mode
    * @return
    */
  private def buildContribEncoder(
                                   df: DataFrame,
                                   columnName: String
                                 ): ExpressionEncoder[Row] = {
    val newSchema = getContribSchema(df, columnName)
    RowEncoder.apply(newSchema)
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

    val featureImportanceDF = dataset.sqlContext
      .table($(featureImportanceView))
      .select("Feature_Index", "Feature", "Importance")
      .orderBy("Feature_Index")
      .collect()

    val featureIndexImportance = SortedMap(
      featureImportanceDF
        .map(
          row =>
            row.getAs[Long](0) -> (row.getAs[String](1), row.getAs[Double](2))
        ): _*
    )

    val predictionsDf = dataset.sqlContext.table($(predictionView))

    val predictionsWithPathsDf =
      pathGenerator(
        predictionsDf,
        featureIndexImportance
      )

    val model = loadModel()

    val contributionsDF = calculateContributions(
      predictionsWithPathsDf,
      featureIndexImportance,
      model
    )
    val featureCount = featureIndexImportance.size
    val contrib_intercept = model.predict(Vectors.dense(Array.fill(featureCount)(0.0)))

    val finalDF =
      contributionsDF.withColumn(
        "contrib_intercept",
        lit(contrib_intercept)
      )
    val finalColRenamedDF =
      if (getDropPathColumn)
        finalDF.transform(appendLabelToColumnNames(getLabel)).drop("paths")
      else finalDF.transform(appendLabelToColumnNames(getLabel))

    finalColRenamedDF.createOrReplaceTempView(getPredictionView)

    finalColRenamedDF
  }


  private def loadModel(): PredictionModel[Vector, _] = {
    val model =
      if (getIsClassification) {
        getEnsembleType.toLowerCase() match {
          case "dct" => DecisionTreeRegressionModel.load(getModelPath)
          case "gbt" => GBTRegressionModel.load(getModelPath)
          case "rf" => RandomForestClassificationModel.load(getModelPath)
          case "xgboost4j" => XGBoostClassificationModel.load(getModelPath).setAllowZeroForMissingValue(true)
        }
      } else {
        getEnsembleType toLowerCase() match {
          case "dct" => DecisionTreeClassificationModel.load(getModelPath)
          case "gbt" => GBTClassificationModel.load(getModelPath)
          case "rf" => RandomForestRegressionModel.load(getModelPath)
          case "xgboost4j" => XGBoostRegressionModel.load(getModelPath).setAllowZeroForMissingValue(true)
        }
      }
    model
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
      List("contrib", "contrib_intercept", "contrib_sum")
    val filteredColumns = df.columns.filter(x => contribColumns.contains(x))
    filteredColumns.foldLeft(df) { (memoDF, colName) =>
      memoDF.withColumnRenamed(colName, s"prediction_${label}_${colName}")
    }
  }

  /**
    * This is the main entry point to calculate linear contribution of each feature
    *
    * @param df
    * @param featureIndexImportance
    * @return
    */
  private def pathGenerator(
                             df: DataFrame,
                             featureIndexImportance: SortedMap[Long, (String, Double)]
                           ): DataFrame = {
    val encoder =
      buildPathsEncoder(df, "paths")
    val func =
      pathGeneratorRow(df.schema)(
        featureIndexImportance
      )

    df.mapPartitions(x => x.map(func))(encoder)
  }

  /*
    Map over Rows and feature to calculate inclusion and exclusion tree path
    ----------------------------------------------------------------------
   */
  private val pathGeneratorRow
  : (StructType) => (SortedMap[Long, (String, Double)]) => Row => Row =
    (schema) =>
      (featureIndexImportance) =>
        (row) => {
          val calculatedPaths = featureIndexImportance.map {
            case (outerFeatureNum, (outerFeatureName, outerImportance)) =>
              // 4
              val outerFeatureVal =
                row
                  .get(schema.fieldIndex(outerFeatureName))
                  .toString
                  .toDouble
              if (outerFeatureVal == 0) {
                val featureCount = featureIndexImportance.size
                // handle feature has no contribution
                outerFeatureNum -> Row(
                  0L,
                  Vectors.dense(Array.fill(featureCount)(0.0)),
                  Vectors.dense(Array.fill(featureCount)(0.0))
                )
              } else {
                // handle exclusion
                val exclusionPath = featureIndexImportance.map {
                  case (_, (innerFeatureName, innerImportance)) =>
                    val innerFeatureVal =
                      row
                        .get(schema.fieldIndex(innerFeatureName))
                        .toString
                        .toDouble
                    if (innerImportance <= outerImportance) 0.0
                    else innerFeatureVal
                }.toArray
                // handle inclusion
                val inclusionPath = featureIndexImportance.map {
                  case (_, (innerFeatureName, innerImportance)) =>
                    val innerFeatureVal =
                      row
                        .get(schema.fieldIndex(innerFeatureName))
                        .toString
                        .toDouble
                    if (innerImportance < outerImportance) 0.0
                    else innerFeatureVal
                }.toArray

                outerFeatureNum -> Row(
                  1L,
                  Vectors.dense(inclusionPath),
                  Vectors.dense(exclusionPath)
                )
              }
          }
          Row.merge(row, Row(calculatedPaths))
        }

  private def calculateContributions(
                                      df: DataFrame,
                                      featureIndexImportance: SortedMap[Long, (String, Double)],
                                      model: PredictionModel[Vector, _]
                                    ): DataFrame = {
    val encoder =
      buildContribEncoder(df, "contrib")
    val func =
      contributionsRows(df.schema)(featureIndexImportance, model)
    df.mapPartitions(x => x.map(func))(encoder)
  }

  /*
     Map over Rows and feature to calculate contributions
     ----------------------------------------------------------------------
   */
  private val contributionsRows: StructType => (
    SortedMap[Long, (String, Double)],
      PredictionModel[Vector, _]
    ) => Row => Row =
    (schema) =>
      (featureIndexImportance, model) =>
        (row) => {
          val path = row.getMap[Long, Row](schema.fieldIndex("paths"))
          val contributions: Seq[Double] = featureIndexImportance.map {
            case (outerFeatureNum, _) =>
              path.get(outerFeatureNum) match {
                case Some(
                Row(
                _,
                inclusionVector: Vector,
                exclusionVector: Vector
                )
                ) =>
                  val contrib =
                    model.predict(inclusionVector) - model.predict(exclusionVector)
                  contrib
              }
          }.toSeq
          Row.merge(
            row,
            Row.fromSeq(
              Seq(
                contributions,
                contributions.sum,
                Vectors.dense(contributions.toArray)
              )
            )
          )
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

object EnsembleTreeExplainTransformer
  extends DefaultParamsReadable[EnsembleTreeExplainTransformer] {
  override def load(path: String): EnsembleTreeExplainTransformer =
    super.load(path)
}
