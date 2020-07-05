package com.clarifyhealth.ohe.decoder

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.functions.{col, expr}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset}

class OneHotDecoder(override val uid: String)
  extends Transformer
    with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("OneHotDecoder"))

  // Transformer Params
  // Defining a Param requires 3 elements:
  //  - Param definition
  //  - Param getter method
  //  - Param setter method
  // (The getter and setter are technically not required, but they are nice standards to follow.)

  /**
    * Param for oheSuffix.
    */
  final val oheSuffix: Param[String] =
    new Param[String](
      this,
      "oheSuffix",
      "input oheSuffix name"
    )

  final def getOheSuffix: String = $(oheSuffix)

  final def setOheSuffix(value: String): OneHotDecoder =
    set(oheSuffix, value)

  /**
    * Param for oheSuffix.
    */
  final val idxSuffix: Param[String] =
    new Param[String](
      this,
      "idxSuffix",
      "input idxSuffix name"
    )

  final def getIdxSuffix: String = $(idxSuffix)

  final def setIdxSuffix(value: String): OneHotDecoder =
    set(idxSuffix, value)

  /**
    * Param for oheSuffix.
    */
  final val unknownSuffix: Param[String] =
    new Param[String](
      this,
      "unknownSuffix",
      "input unknownSuffix name"
    )

  final def getUnknownSuffix: String = $(unknownSuffix)

  final def setUnknownSuffix(value: String): OneHotDecoder =
    set(unknownSuffix, value)

  // (Optional) You can set defaults for Param values if you like.
  setDefault(
    oheSuffix -> "_OHE",
    idxSuffix -> "_IDX",
    unknownSuffix -> "Unknown"
  )

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

    val datasetDF = dataset.toDF()
    val oheColumns = getOHEColumns(datasetDF)

    val oheDecodedColumns = oheColumns.flatMap { encodedColumn =>
      val metadata: Array[Metadata] = getEncodingMapping(datasetDF, encodedColumn)
      metadata.map { item =>
        val idx = item.getLong("idx")
        val name = item.getString("name")
        val indexColumn = getIDXColumn(encodedColumn)
        val decodedColumn = getDecodedColumn(encodedColumn, name)
        expr(s"cast(${indexColumn}=${idx} as integer)").alias(decodedColumn)
      }
    }
    datasetDF.select(Array(col("*")) ++ oheDecodedColumns: _*)
  }

  def getOHEColumns(df: DataFrame): Array[String] = {
    df.columns.filter(x => x.endsWith(getOheSuffix))
  }

  def getIDXColumn(encodedColumn: String): String = {
    encodedColumn.replace(getOheSuffix, getIdxSuffix)
  }

  def getDecodedColumn(encodedColumn: String, value: String): String = {
    val cleanedValue = value.replaceAll("[^0-9a-zA-Z]+", "_")
    s"${encodedColumn}_${cleanedValue}"
  }

  def getEncodingMapping(df: DataFrame, encoded_column: String): Array[Metadata] = {
    val column_meta = df.schema(encoded_column).metadata
    val mlAttrData = column_meta.getMetadata("ml_attr")
    val attrsData = mlAttrData.getMetadata("attrs")
    val encodedMapping = attrsData.getMetadataArray("binary")
    encodedMapping
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


