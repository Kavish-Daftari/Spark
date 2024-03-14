// Databricks notebook source
import org.apache.spark.sql.DataFrame

// Load train and test datasets
val trainData: DataFrame = spark.read.option("header", "true").option("inferSchema", "true").csv("/FileStore/tables/train.csv")
val testData: DataFrame = spark.read.option("header", "true").option("inferSchema", "true").csv("/FileStore/tables/test.csv")

// Display the first few rows of the training data
trainData.show()


// COMMAND ----------

// Display summary statistics for numeric columns
trainData.describe().show()

// Count of missing values per column
val cols = trainData.columns
cols.foreach(col => {
  val missingCount = trainData.filter(trainData(col).isNull || trainData(col) === "" || trainData(col).isNaN).count()
  println(s"Missing values in $col: $missingCount")
})

// Survival rate by different features
trainData.groupBy("Sex").agg(avg("Survived")).show()
trainData.groupBy("Pclass").agg(avg("Survived")).show()
trainData.groupBy("Embarked").agg(avg("Survived")).show()


// COMMAND ----------

import org.apache.spark.ml.feature.{StringIndexer, OneHotEncoder, VectorAssembler, Imputer}

// Handle missing values
val imputer = new Imputer().setInputCols(Array("Age", "Fare")).setOutputCols(Array("AgeImputed", "FareImputed")).setStrategy("median")

// Encoding categorical variables
val sexIndexer = new StringIndexer().setInputCol("Sex").setOutputCol("SexIndex").setHandleInvalid("keep")
val embarkedIndexer = new StringIndexer().setInputCol("Embarked").setOutputCol("EmbarkedIndex").setHandleInvalid("keep")

val sexEncoder = new OneHotEncoder().setInputCol("SexIndex").setOutputCol("SexVec")
val embarkedEncoder = new OneHotEncoder().setInputCol("EmbarkedIndex").setOutputCol("EmbarkedVec")

// Assemble features into a vector
val assembler = new VectorAssembler().setInputCols(Array("Pclass", "SexVec", "AgeImputed", "FareImputed", "EmbarkedVec")).setOutputCol("features").setHandleInvalid("skip")

// Pipeline for preprocessing
val pipeline = new Pipeline().setStages(Array(imputer, sexIndexer, embarkedIndexer, sexEncoder, embarkedEncoder, assembler))

// Fit and transform the training data
val trainDataPrepared = pipeline.fit(trainData).transform(trainData)


// COMMAND ----------

import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.{StringIndexer, OneHotEncoder, VectorAssembler, Imputer}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.DataFrame

// Split the original training data into new training and validation sets
val Array(newTrainingData, validationData) = trainData.randomSplit(Array(0.8, 0.2), seed = 42)

// Preprocess the data
val imputer = new Imputer().setInputCols(Array("Age", "Fare")).setOutputCols(Array("AgeImputed", "FareImputed")).setStrategy("median")
val sexIndexer = new StringIndexer().setInputCol("Sex").setOutputCol("SexIndex").setHandleInvalid("keep")
val embarkedIndexer = new StringIndexer().setInputCol("Embarked").setOutputCol("EmbarkedIndex").setHandleInvalid("keep")
val sexEncoder = new OneHotEncoder().setInputCol("SexIndex").setOutputCol("SexVec")
val embarkedEncoder = new OneHotEncoder().setInputCol("EmbarkedIndex").setOutputCol("EmbarkedVec")
val assembler = new VectorAssembler().setInputCols(Array("Pclass", "SexVec", "AgeImputed", "FareImputed", "EmbarkedVec")).setOutputCol("features")

// RandomForestClassifier
val rf = new RandomForestClassifier().setLabelCol("Survived").setFeaturesCol("features").setNumTrees(10)

// Construct the Pipeline
val pipeline = new Pipeline().setStages(Array(imputer, sexIndexer, embarkedIndexer, sexEncoder, embarkedEncoder, assembler, rf))

// Train the model
val model: PipelineModel = pipeline.fit(newTrainingData)

// Make predictions on the validation set
val predictions: DataFrame = model.transform(validationData)

// Calculate accuracy manually
val correctPredictions = predictions.withColumn("Correct", expr("prediction == Survived"))
val accuracy = correctPredictions.agg(avg(expr("cast(Correct as int)"))).first().getDouble(0)

println(s"Model accuracy: $accuracy")

