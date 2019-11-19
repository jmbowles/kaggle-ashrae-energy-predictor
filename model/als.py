from __future__ import print_function
"""
0: electricity, 1: chilledwater, 2: steam, 3: hotwater

spark-submit --driver-memory=20g --conf spark.dynamicAllocation.enabled=true --conf spark.shuffle.service.enabled=true
"""
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import Imputer, Bucketizer
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType

spark = SparkSession.builder.appName("ALS Training") \
	.enableHiveSupport() \
	.getOrCreate()


def get_building(df, building_id):
	
	return df.where(F.expr("building_id = {0}".format(building_id)))

def get_meter(df, meter):
	
	return df.where(F.expr("meter = {0}".format(meter)))

def get_meters(df):

	return df.select("meter").distinct().orderBy("meter")
	
def get_buildings(building_id=None):

	df = spark.read.load("../datasets/building_metadata.csv", format="csv", sep=",", inferSchema="true", header="true").select("building_id").orderBy("building_id")

	if building_id:
		return df.where(df.building_id >= building_id).orderBy("building_id")
	else:
		return df
		
def fit(df):

	day_splits = [-float("inf"), 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, float("inf")]
	bucketizer = Bucketizer(splits=day_splits, inputCol="day", outputCol="bucket", handleInvalid="error")
	
	als = ALS(userCol="month", itemCol="bucket", ratingCol="meter_reading")
	pipeline = Pipeline(stages=[bucketizer, als])
	evaluator = RegressionEvaluator(labelCol="meter_reading", predictionCol="prediction", metricName="rmse")

	params = ParamGridBuilder() \
				.addGrid(als.rank, [10]) \
				.addGrid(als.maxIter, [10.0]) \
				.addGrid(als.nonnegative, [False]) \
				.build()

	validator = CrossValidator(estimator=pipeline, estimatorParamMaps=params, evaluator=evaluator, numFolds=2, seed=51)
	model = validator.fit(df)

	return model.bestModel

def save_model(building_id, meter, model):

	model_path = "output/als_model_{0}_{1}".format(building_id, meter)
	model.write().overwrite().save(model_path)


print("Loading all data")
df = spark.table("training")
df = df.dropna(how="all", subset="air_temperature")
df = df.withColumn("air_temperature", df.air_temperature.cast("integer"))

print("Caching splits")
training, _ = df.randomSplit([0.8, 0.2])
training.cache()

starting_building = 1293
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





	

