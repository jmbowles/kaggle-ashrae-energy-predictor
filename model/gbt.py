from __future__ import print_function
"""
0: electricity, 1: chilledwater, 2: steam, 3: hotwater

spark-submit --driver-memory=20g --conf spark.dynamicAllocation.enabled=true --conf spark.shuffle.service.enabled=true
"""
import math

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, Imputer
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType

spark = SparkSession.builder.appName("GBT Regression Training") \
	.enableHiveSupport() \
	.getOrCreate()


def get_building(df, building_id):

	return df.where(F.expr("building_id = {0}".format(building_id)))

def get_buildings(building_id=None):

	df = spark.read.load("../datasets/building_metadata.csv", format="csv", sep=",", inferSchema="true", header="true").select("building_id").orderBy("building_id")

	if building_id:
		return df.where(df.building_id >= building_id).orderBy("building_id")
	else:
		return df

def get_meter(df, meter):
	
	return df.where(F.expr("meter = {0}".format(meter)))

def get_meters(df):

	return df.select("meter").distinct().orderBy("meter")

def fit(df):

	cols = ["air_temperature_est", "dew_temperature_est", "wind_speed_est", "month", "day", "hour"]
	imputer = Imputer(strategy="median", inputCols=["air_temperature", "dew_temperature", "wind_speed"], outputCols=["air_temperature_est", "dew_temperature_est", "wind_speed_est"])
	vector = VectorAssembler(inputCols=cols, outputCol="features", handleInvalid="error")
	regression = GBTRegressor(featuresCol="features", labelCol="meter_reading", predictionCol="prediction")
	pipeline = Pipeline(stages=[imputer, vector, regression])
	evaluator = RegressionEvaluator(labelCol="meter_reading", predictionCol="prediction", metricName="rmse")

	params = ParamGridBuilder() \
				.addGrid(imputer.strategy, ["mean", "median"]) \
				.addGrid(regression.maxIter, [30]) \
				.build()

	validator = CrossValidator(estimator=pipeline, estimatorParamMaps=params, evaluator=evaluator, numFolds=3, seed=51)
	model = validator.fit(df)

	return model.bestModel

def save_model(building_id, meter, model):

	model_path = "output/gbt_model_{0}_{1}".format(building_id, meter)
	model.write().overwrite().save(model_path)


print("Loading all data")
training = spark.table("training")
training.cache()

starting_building = 1424
buildings = get_buildings(starting_building)

for row in buildings.toLocalIterator():
	
	building_id = row.building_id

	print("Filtering training data for building: {0}".format(building_id))
	building = get_building(training, building_id)

	meters = get_meters(building)

	for row in meters.toLocalIterator():

		meter_id = row.meter
		building_meter = get_meter(building, meter_id)
		
		print("Applying fit for building: {0}, meter {1}".format(building_id, meter_id))
		model = fit(building_meter)

		print("Saving model")
		save_model(building_id, meter_id, model)




	

