name := "spark_model_explainer"
organization := "clarifyhealth"

version := "0.1"

scalaVersion := "2.11.12"

val sparkVersion = "2.4.5"
val xgboost4jVersion = "1.0.0"

scalacOptions := Seq("-unchecked", "-deprecation")

// turn off parallel tests
parallelExecution in Test := false
// remove version-specific scala dirs
crossPaths := false

developers := List(
  Developer("iman", "Iman Haji", "@iman", url("https://github.com/")),
  Developer("imranq2", "Imran Qureshi", "@imranq2", url("https://github.com/imranq2")),
  Developer("alvinhenrick", "Alvin Henrick", "@alvinhenrick", url("https://github.com/alvinhenrick")),
)

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion % Provided,
  "org.apache.spark" %% "spark-sql" % sparkVersion % Provided,
  "org.apache.spark" %% "spark-catalyst" % sparkVersion % Provided,
  "org.apache.spark" %% "spark-mllib" % sparkVersion % Provided,
  "org.apache.spark" %% "spark-core" % sparkVersion % "test" classifier "tests",
  "org.apache.spark" %% "spark-sql" % sparkVersion % "test" classifier "tests",
  "org.apache.spark" %% "spark-catalyst" % sparkVersion % "test" classifier "tests",
  "org.apache.spark" %% "spark-mllib" % sparkVersion % "test" classifier "tests",
  "ml.dmlc" %% "xgboost4j" % xgboost4jVersion,
  "ml.dmlc" %% "xgboost4j-spark" % xgboost4jVersion,
  "org.scalatest" %% "scalatest" % "3.0.8" % Test,
  "org.scalactic" %% "scalactic" % "3.0.8" % Test,

)

excludeDependencies += "com.esotericsoftware.kryo" % "kryo"

