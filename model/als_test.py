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


def get_building(df, building_id, meter):
	
	return df.where(F.expr("building_id = {0} and meter = {1}".format(building_id, meter)))

def get_buildings(building_id=None):

	df = spark.read.load("../datasets/building_metadata.csv", format="csv", sep=",", inferSchema="true", header="true").select("building_id").orderBy("building_id")

	if building_id:
		return df.where(df.building_id == building_id).orderBy("building_id")
	else:
		return df

def fit(df):

	#temperature_splits = [-float("inf"), -23.0, -18.0, -13.0, -8.0, -3.0, 2.0, 7.0, 12.0, 17.0, 22.0, 27.0, 32.0, 37.0, float("inf")]
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

def predict(model, test, building_id):
	
	predictions = model.transform(test)
	predictions = predictions.withColumn("prediction", F.when(predictions.prediction < 0, F.lit(0.0)).otherwise(predictions.prediction))
	
	evaluator = RegressionEvaluator(labelCol="meter_reading", predictionCol="prediction", metricName="rmse")
	rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
	r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})
	mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})

	return (predictions, (rmse, r2, mae))

def save_model(building_id, model):

	model_path = "output/als_model_{0}".format(building_id)
	model.write().overwrite().save(model_path)

def save_existing_predictions(building_id):

	if not None:
		temp = spark.sql("select * from als_predictions where building_id < {0}".format(building_id))
		temp.coalesce(1).write.saveAsTable("als_predictions_temp", format="parquet", mode="append")
		spark.sql("drop table als_predictions")
		spark.sql("alter table als_predictions_temp rename to als_predictions")


print("Loading all data")
df = spark.table("training")
#df = df.dropna(thresh=0, subset="meter_reading")
df = df.dropna(how="all", subset="air_temperature")
df = df.withColumn("air_temperature", df.air_temperature.cast("integer"))

print("Dropping tables")
spark.sql("drop table if exists als_predictions")
spark.sql("drop table if exists als_predictions_metrics")

print("Caching splits")
training, test = df.randomSplit([0.8, 0.2])
training.cache()
test.cache()

#training.crosstab("building_id", "meter").withColumnRenamed("building_id_meter", "building_id").withColumn("building_id", F.col("building_id").cast("integer")).orderBy("building_id").show()

starting_building = 1352
meter = 1
#save_existing_predictions(starting_building)
buildings = get_buildings(starting_building)

for row in buildings.toLocalIterator():
	
	building_id = row.building_id

	print("Filtering training data for building: {0}".format(building_id))
	building = get_building(training, building_id, meter)

	print("Applying fit for building: {0}".format(building_id))
	model = fit(building)

	print("Saving model")
	save_model(building_id, model)

	print("Filtering test data for building: {0}".format(building_id))
	test_building = get_building(test, building_id, meter)

	print("Predicting test data")
	prediction = predict(model, test_building, building_id)

	print("Saving predictions")
	prediction, metrics = prediction
	prediction.coalesce(1).write.saveAsTable("als_predictions", format="parquet", mode="append")

	print("Saving metrics")

	schema = StructType([StructField("building_id", IntegerType(), False), 
				StructField("rmse", DoubleType(), False),
				StructField("r2", DoubleType(), False),
				StructField("mae", DoubleType(), False)])

	rmse, r2, mae = metrics
	print("RMSE, R2, MAE: {0}, {1}, {2}".format(rmse, r2, mae))
	metric = spark.createDataFrame([(building_id, rmse, r2, mae)], schema)
	metric.coalesce(1).write.saveAsTable("als_predictions_metrics", format="parquet", mode="append")


cols = ["timestamp", "building_id", "meter", "meter_reading", "prediction", "log_squared_error"]

print("Predictions")
p = spark.table("als_predictions")
p = p.withColumn("log_squared_error", F.pow(F.log(p.prediction + 1) - F.log(p.meter_reading + 1), 2)).select(*cols).orderBy("building_id", "timestamp")
p.show()

import math

log_squared_error = p.agg(F.sum("log_squared_error").alias("lse")).collect()[0]["lse"]
rmsle = math.sqrt(log_squared_error / p.count())
print("RMSLE: {0}".format(rmsle))





	

