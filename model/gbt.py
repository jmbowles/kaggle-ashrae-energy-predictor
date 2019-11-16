from __future__ import print_function
"""
0: electricity, 1: chilledwater, 2: steam, 3: hotwater

spark-submit --driver-memory=20g --conf spark.dynamicAllocation.enabled=true --conf spark.shuffle.service.enabled=true
"""
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, Imputer, PolynomialExpansion
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

def fit(df):

	cols = ["air_temperature_est", "dew_temperature_est", "wind_speed_est", "meter", "month", "day", "hour"]
	imputer = Imputer(strategy="median", inputCols=["air_temperature", "dew_temperature", "wind_speed"], outputCols=["air_temperature_est", "dew_temperature_est", "wind_speed_est"])
	vector = VectorAssembler(inputCols=cols, outputCol="vector", handleInvalid="error")
	poly = PolynomialExpansion(degree=3, inputCol="vector", outputCol="features")
	regression = GBTRegressor(featuresCol="features", labelCol="meter_reading", predictionCol="prediction")
	pipeline = Pipeline(stages=[imputer, vector, poly, regression])
	evaluator = RegressionEvaluator(labelCol="meter_reading", predictionCol="prediction", metricName="rmse")

	params = ParamGridBuilder() \
				.addGrid(imputer.strategy, ["mean", "median"]) \
				.addGrid(poly.degree, [2]) \
				.addGrid(regression.maxIter, [30]) \
				.build()

	validator = CrossValidator(estimator=pipeline, estimatorParamMaps=params, evaluator=evaluator, numFolds=3, seed=51)
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

	model_path = "output/gbt_model_{0}".format(building_id)
	model.write().overwrite().save(model_path)

def save_existing_predictions(building_id):

	temp = spark.sql("select * from gbt_predictions where building_id < {0}".format(building_id))
	temp.coalesce(1).write.saveAsTable("gbt_predictions_temp", format="parquet", mode="append")
	spark.sql("drop table gbt_predictions")
	spark.sql("alter table gbt_predictions_temp rename to gbt_predictions")


print("Loading all data")
df = spark.table("training")
df = df.withColumn("meter_reading", F.when(df.meter_reading == 0, F.lit(1.0)).otherwise(df.meter_reading))

#print("Dropping tables")
#spark.sql("drop table if exists gbt_predictions")
#spark.sql("drop table if exists gbt_predictions_metrics")

print("Caching splits")
training, test = df.randomSplit([0.8, 0.2])
training.cache()
test.cache()

starting_building = 1020
save_existing_predictions(starting_building)
buildings = get_buildings(starting_building)

for row in buildings.toLocalIterator():
	
	building_id = row.building_id

	print("Filtering training data for building: {0}".format(building_id))
	building = get_building(training, building_id)

	print("Applying fit for building: {0}".format(building_id))
	model = fit(building)

	print("Saving model")
	save_model(building_id, model)

	print("Filtering test data for building: {0}".format(building_id))
	test_building = get_building(test, building_id)

	print("Predicting test data")
	prediction = predict(model, test_building, building_id)

	print("Saving predictions")
	prediction, metrics = prediction
	#prediction.select("timestamp", "meter_reading", "prediction").show()
	prediction.coalesce(1).write.saveAsTable("gbt_predictions", format="parquet", mode="append")

	print("Saving metrics")

	schema = StructType([StructField("building_id", IntegerType(), False), 
				StructField("rmse", DoubleType(), False),
				StructField("r2", DoubleType(), False),
				StructField("mae", DoubleType(), False)])

	rmse, r2, mae = metrics
	print("RMSE, R2, MAE: {0}, {1}, {2}".format(rmse, r2, mae))
	metric = spark.createDataFrame([(building_id, rmse, r2, mae)], schema)
	metric.coalesce(1).write.saveAsTable("gbt_predictions_metrics", format="parquet", mode="append")


cols ["timestamp", "building_id", "air_temperature", "dew_temperature", "meter", "meter_reading", "prediction", "log_squared_error"]

p = spark.table("gbt_predictions")
p = p.withColumn("log_squared_error", F.pow(F.log(p.prediction + 1) - F.log(p.meter_reading + 1), 2)).select(*cols).orderBy("building_id", "timestamp")
p.show()

import math

log_squared_error = p.agg(F.sum("log_squared_error").alias("lse")).collect()[0]["lse"]
rmsle = math.sqrt(log_squared_error / p.count())
print("RMSLE: {0}".format(rmsle))





	

