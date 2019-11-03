from __future__ import print_function
"""
Bayes Rule:

P(Malware | X) = P(X | Malware) P(Malware) / P(X)

Where X = 83 column values in each dataset:

est set accuracy = 0.64626646587
"""
import pickle

from pyspark.sql import Window
from pyspark.ml import Pipeline
from pyspark.ml.feature import FeatureHasher, VectorAssembler, MinMaxScaler, PolynomialExpansion
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

import pyspark.sql.functions as F
from pyspark.sql import SparkSession

pipeline_model_path = "output/naive_bayes_pipeline_model"

spark = SparkSession.builder.appName("NaiveBayes") \
	.config("spark.dynamicAllocation.enabled", "true") \
	.config("spark.shuffle.service.enabled", "true") \
	.config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
	.enableHiveSupport() \
	.getOrCreate()

print("Loading and Caching Data")
df = spark.read.table("training")
df = df.withColumn("elapsed_days", F.datediff(df.OSVersionUTC, df.AvSigVersionUTC))

continuous_cols = ["elapsed_days"] 
#derived_cols = [col for col in df.columns if col.startswith("Derived_")]
#continuous_cols = continuous_cols + derived_cols
categorical_cols = ["SmartScreen", "Wdft_RegionIdentifier", "CountryIdentifier", "AVProductStatesIdentifier"]
feature_cols = ["categorical_features", "continuous_features"] 

meta_cols = ["HasDetections", "MachineIdentifier"]
drop_cols = list(set(df.columns) - set(meta_cols) - set(continuous_cols) - set(categorical_cols) - set(feature_cols))
df = df.drop(*drop_cols)
df = df.cache()

print("Creating Splits")
train, test = df.randomSplit([0.8, 0.2])

print("Selected Features Count: {0}".format(len(feature_cols)))
print("Selected Features: {0}".format(feature_cols))

print("Building Pipeline")
categorical_hasher = FeatureHasher(inputCols=categorical_cols, outputCol="categorical_features", categoricalCols=categorical_cols)
continuous_vector = VectorAssembler(inputCols=continuous_cols, outputCol="continuous_vector")
scaler = MinMaxScaler(min=0.0, max=1.0, inputCol="continuous_vector", outputCol="continuous_features")
features = VectorAssembler(inputCols=feature_cols, outputCol="features")
bayes = NaiveBayes(smoothing=1.0, featuresCol="features", labelCol="HasDetections", predictionCol="prediction", modelType="multinomial")
pipeline = Pipeline(stages=[categorical_hasher, continuous_vector, scaler, features, bayes])
evaluator = MulticlassClassificationEvaluator(labelCol="HasDetections", predictionCol="prediction", metricName="f1")

print("Configuring CrossValidation")
params = ParamGridBuilder() \
			.addGrid(categorical_hasher.numFeatures, [32768]) \
			.build()

validator = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=params,
                          evaluator=evaluator,
                          numFolds=5)

print("Fitting -> Training Data")
pipeline_model = validator.fit(train)

print("Fitting -> Test Data")
predictions = pipeline_model.transform(test)
predictions.select("HasDetections", "MachineIdentifier", "probability", "prediction").show(truncate=False)

print("Computing Accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test set accuracy = {0}".format(accuracy))

print("Saving Pipeline Model")
pipeline_model.bestModel.write().overwrite().save(pipeline_model_path)

print("Saving Predictions")
predictions.coalesce(5).write.saveAsTable("naive_bayes_predictions", format="parquet", mode="overwrite")






