from __future__ import print_function
"""

"""
import pickle

from pyspark.sql import Window
from pyspark.ml import Pipeline
from pyspark.ml.feature import FeatureHasher, VectorAssembler, MinMaxScaler, PolynomialExpansion
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

import pyspark.sql.functions as F
from pyspark.sql import SparkSession

pipeline_model_path = "output/logistic_pipeline_model"

spark = SparkSession.builder.appName("LogisticRegression") \
	.config("spark.dynamicAllocation.enabled", "true") \
	.config("spark.shuffle.service.enabled", "true") \
	.config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
	.enableHiveSupport() \
	.getOrCreate()

print("Loading and Caching Data")
df = spark.read.table("training")
df = df.withColumn("elapsed_days", F.datediff(df.OSVersionUTC, df.AvSigVersionUTC))
 
continuous_cols = ["elapsed_days", "AVProductsInstalled", "Census_InternalPrimaryDisplayResolutionVertical", "Census_IsTouchEnabled", "Census_TotalPhysicalRAM", "Census_ProcessorCoreCount"]
categorical_cols = ["SmartScreen", "Wdft_RegionIdentifier", "CountryIdentifier", "AVProductStatesIdentifier", "Wdft_IsGamer", "Census_PowerPlatformRoleName"]
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
scaler = MinMaxScaler(min=0.0, max=100.0, inputCol="continuous_vector", outputCol="continuous_features")
features = VectorAssembler(inputCols=feature_cols, outputCol="features")
regression = LogisticRegression(featuresCol=features.getOutputCol(), labelCol="HasDetections", regParam=0.0, elasticNetParam=0.0, tol=1e-06, threshold=0.5, family="auto")
pipeline = Pipeline(stages=[categorical_hasher, continuous_vector, scaler, features, regression])
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
predictions.coalesce(5).write.saveAsTable("logistic_predictions", format="parquet", mode="overwrite")






