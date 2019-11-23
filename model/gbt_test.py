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
		return df.where(df.building_id == building_id).orderBy("building_id")
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

def predict(model, test):
	
	predictions = model.transform(test)
	predictions = predictions.withColumn("prediction", F.when(predictions.prediction < 0, F.lit(0.0)).otherwise(predictions.prediction))
	
	evaluator = RegressionEvaluator(labelCol="meter_reading", predictionCol="prediction", metricName="rmse")
	rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
	r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})
	mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})

	return (predictions, (rmse, r2, mae))

def save_model(building_id, meter, model):

	model_path = "output/gbt_model_{0}_{1}".format(building_id, meter)
	model.write().overwrite().save(model_path)

def save_existing_predictions(building_id):

	temp = spark.sql("select * from gbt_predictions where building_id < {0}".format(building_id))
	temp.coalesce(1).write.saveAsTable("gbt_predictions_temp", format="parquet", mode="append")
	spark.sql("drop table gbt_predictions")
	spark.sql("alter table gbt_predictions_temp rename to gbt_predictions")


metrics_schema = StructType([StructField("building_id", IntegerType(), False), 
							StructField("rmse", DoubleType(), False),
							StructField("r2", DoubleType(), False),
							StructField("mae", DoubleType(), False),
							StructField("rmsle", DoubleType(), False)])

print("Loading all data")
df = spark.table("training")

print("Dropping tables")
spark.sql("drop table if exists gbt_predictions")
spark.sql("drop table if exists gbt_predictions_metrics")

print("Caching splits")
training, test = df.randomSplit([0.8, 0.2])
#training = training.where(training.meter_reading > 0)

training.cache()
test.cache()

starting_building = 1322
#save_existing_predictions(starting_building)
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

		print("Filtering test data for building: {0}, meter {1}".format(building_id, meter_id))
		test_building = get_building(test, building_id)
		building_meter_test = get_meter(test_building, meter_id)

		print("Predicting test data")
		prediction = predict(model, building_meter_test)

		print("Saving predictions")
		prediction, metrics = prediction
		#prediction.select("timestamp", "meter_reading", "prediction").show()
		prediction.coalesce(1).write.saveAsTable("gbt_predictions", format="parquet", mode="append")

		print("Saving metrics")
		prediction = prediction.withColumn("log_squared_error", F.pow(F.log(prediction.prediction + 1) - F.log(prediction.meter_reading + 1), 2))
		log_squared_error = prediction.agg(F.sum("log_squared_error").alias("lse")).collect()[0]["lse"]
		rmsle = math.sqrt(log_squared_error / prediction.count())

		rmse, r2, mae = metrics
		print("RMSE, R2, MAE, RMSLE: {0}, {1}, {2}, {3}".format(rmse, r2, mae, rmsle))
		metric = spark.createDataFrame([(building_id, rmse, r2, mae, rmsle)], metrics_schema)
		metric.coalesce(1).write.saveAsTable("gbt_predictions_metrics", format="parquet", mode="append")
		

"""
RMSLE = 0.88 is based upon 939 / 1449 buildings (buildings 510 - 1448, zero-based index). Made a mistake by dropping gbt_predictions
for buildings 0 - 509 after restarting job due to memory issues
"""
cols = ["timestamp", "building_id", "meter", "meter_reading", "prediction", "log_squared_error"]

p = spark.table("gbt_predictions")
p = p.withColumn("log_squared_error", F.pow(F.log(p.prediction + 1) - F.log(p.meter_reading + 1), 2)).select(*cols).orderBy("building_id", "timestamp")
p.show()

log_squared_error = p.agg(F.sum("log_squared_error").alias("lse")).collect()[0]["lse"]
rmsle = math.sqrt(log_squared_error / p.count())
print("Overall RMSLE: {0}".format(rmsle))





	

