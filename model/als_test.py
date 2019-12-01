from __future__ import print_function
"""
0: electricity, 1: chilledwater, 2: steam, 3: hotwater

spark-submit --driver-memory=20g --conf spark.dynamicAllocation.enabled=true --conf spark.shuffle.service.enabled=true
"""
import math

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


def get_meter(df, meter):
	
	return df.where(F.expr("meter = {0}".format(meter)))

def get_meters(df):

	return df.select("meter").distinct().orderBy("meter")

def get_building(df, building_id):
	
	return df.where(F.expr("building_id = {0}".format(building_id)))

def get_buildings(building_id=None):

	df = spark.read.load("../datasets/building_metadata.csv", format="csv", sep=",", inferSchema="true", header="true").select("building_id").orderBy("building_id")

	if building_id:
		return df.where(df.building_id == building_id).orderBy("building_id")
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

	return imputed

def fit(df):
	
	als = ALS(userCol="month", itemCol="day", ratingCol="meter_reading", coldStartStrategy="nan")
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

def predict(model, test):
	
	predictions = model.transform(test)
	predictions = predictions.withColumn("prediction", F.when(predictions.prediction < 0, F.lit(0.0)).otherwise(predictions.prediction))
	
	evaluator = RegressionEvaluator(labelCol="meter_reading", predictionCol="prediction", metricName="rmse")
	rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
	r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})
	mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})

	return (predictions, (rmse, r2, mae))

def save_model(model, building_id, meter):

	model_path = "output/als_test_model_{0}_{1}".format(building_id, meter)
	model.write().overwrite().save(model_path)

metrics_schema = StructType([StructField("building_id", IntegerType(), False), 
							StructField("rmse", DoubleType(), False),
							StructField("r2", DoubleType(), False),
							StructField("mae", DoubleType(), False),
							StructField("rmsle", DoubleType(), False)])

print("Loading all data")
df = spark.table("training")
df = df.withColumn("air_temperature", df.air_temperature.cast("double"))

print("Dropping tables")
spark.sql("drop table if exists als_test_predictions")
spark.sql("drop table if exists als_test_predictions_metrics")

print("Caching splits")
training, test = df.randomSplit([0.8, 0.2])
training.cache()
test.cache()

starting_building = 28
buildings = get_buildings(starting_building)

for row in buildings.toLocalIterator():
	
	building_id = row.building_id
	
	print("Filtering training and test data for building: {0}".format(building_id))
	building = get_building(training, building_id)
	test_building = get_building(test, building_id)

	meters = get_meters(building)

	for row in meters.toLocalIterator():

		meter_id = row.meter
		building_meter = get_meter(building, meter_id)
		print("Applying fit for building: {0}, meter {1}".format(building_id, meter_id))
		building_meter = impute(building_meter, building_id, meter_id)
		model = fit(building_meter)

		print("Saving model")
		save_model(model, building_id, meter_id)

		print("Predicting test data")
		building_meter = get_meter(test_building, meter_id)
		building_meter = impute(building_meter, building_id, meter_id)
		prediction = predict(model, building_meter)

		print("Saving predictions")
		prediction, metrics = prediction
		prediction.coalesce(1).write.saveAsTable("als_test_predictions", format="parquet", mode="append")

		print("Saving metrics")
		prediction = prediction.withColumn("log_squared_error", F.pow(F.log(prediction.prediction + 1) - F.log(prediction.meter_reading + 1), 2))
		log_squared_error = prediction.agg(F.sum("log_squared_error").alias("lse")).collect()[0]["lse"]
		rmsle = math.sqrt(log_squared_error / prediction.count())

		rmse, r2, mae = metrics
		print("RMSE, R2, MAE, RMSLE: {0}, {1}, {2}, {3}".format(rmse, r2, mae, rmsle))
		metric = spark.createDataFrame([(building_id, rmse, r2, mae, rmsle)], metrics_schema)
		metric.coalesce(1).write.saveAsTable("als_test_predictions_metrics", format="parquet", mode="append")
	

cols = ["timestamp", "building_id", "meter", "meter_reading", "prediction", "log_squared_error"]

print("Predictions")
p = spark.table("als_test_predictions")
p = p.withColumn("log_squared_error", F.pow(F.log(p.prediction + 1) - F.log(p.meter_reading + 1), 2)).select(*cols).orderBy("building_id", "timestamp")
p.orderBy("timestamp").show()

log_squared_error = p.agg(F.sum("log_squared_error").alias("lse")).collect()[0]["lse"]
rmsle = math.sqrt(log_squared_error / p.count())
print("RMSLE: {0}".format(rmsle))





	

