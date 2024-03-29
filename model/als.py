from __future__ import print_function
"""
0: electricity, 1: chilledwater, 2: steam, 3: hotwater

spark-submit --driver-memory=20g --conf spark.dynamicAllocation.enabled=true --conf spark.shuffle.service.enabled=true
"""
from pyspark.ml import Pipeline
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

import pyspark.sql.functions as F
from pyspark.sql import SparkSession

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
		
def impute(df, building_id, meter):

	time_series = spark.sql("SELECT explode(sequence(to_timestamp('2016-01-01'), to_timestamp('2017-01-02'), interval 1 hour))").withColumnRenamed("col", "timestamp_seq")
	joined = time_series.join(df, [time_series.timestamp_seq == df.timestamp], "left_outer")
	median = df.approxQuantile("meter_reading", [0.5], 0.001)[0]
	imputed = joined.fillna(median, ["meter_reading"])
	imputed = imputed.drop("timestamp")
	imputed = imputed.withColumnRenamed("timestamp_seq", "timestamp")
	imputed = imputed.withColumn("building_id", F.lit(building_id))
	imputed = imputed.withColumn("meter", F.lit(meter))
	imputed = imputed.withColumn("month", F.month(imputed.timestamp))
	imputed = imputed.withColumn("day", F.dayofmonth(imputed.timestamp))
	imputed = imputed.withColumn("hour", F.hour(imputed.timestamp))
	imputed = imputed.withColumn("meter_reading", F.when(imputed.meter_reading == 0, median).otherwise(imputed.meter_reading))
	imputed = imputed.withColumn("meter_reading", F.log1p(imputed.meter_reading))

	return imputed

def fit(df):

	als = ALS(userCol="month", itemCol="day", ratingCol="meter_reading")
	pipeline = Pipeline(stages=[als])
	evaluator = RegressionEvaluator(labelCol="meter_reading", predictionCol="prediction", metricName="rmse")

	params = ParamGridBuilder() \
				.addGrid(als.rank, [10]) \
				.addGrid(als.maxIter, [10.0]) \
				.addGrid(als.nonnegative, [False]) \
				.build()

	validator = CrossValidator(estimator=pipeline, estimatorParamMaps=params, evaluator=evaluator, numFolds=3, parallelism=4, seed=51)
	model = validator.fit(df)

	return model.bestModel

def save_model(model, building_id, meter):

	model_path = "output/als_model_{0}_{1}".format(building_id, meter)
	model.write().overwrite().save(model_path)


print("Loading all data")
df = spark.table("training")
df.cache()

starting_building = 1244
buildings = get_buildings(starting_building)

for row in buildings.toLocalIterator():
	
	building_id = row.building_id

	print("Filtering training data for building: {0}".format(building_id))
	building = get_building(df, building_id)
	meters = get_meters(building)

	for row in meters.toLocalIterator():

		meter_id = row.meter
		building_meter = get_meter(building, meter_id)
		print("Applying fit for building: {0}, meter {1}".format(building_id, meter_id))
		building_meter = impute(building_meter, building_id, meter_id)
		model = fit(building_meter)

		print("Saving model")
		save_model(model, building_id, meter_id)





	

