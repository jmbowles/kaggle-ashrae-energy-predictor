from __future__ import print_function
"""
0: electricity, 1: chilledwater, 2: steam, 3: hotwater

spark-submit --driver-memory=20g --conf spark.dynamicAllocation.enabled=true --conf spark.shuffle.service.enabled=true
"""
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, Imputer, PolynomialExpansion
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType

spark = SparkSession.builder.appName("Linear Regression Training") \
	.config("spark.dynamicAllocation.enabled", "true") \
	.config("spark.shuffle.service.enabled", "true") \
	.config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
	.enableHiveSupport() \
	.getOrCreate()

def remove_outliers(df):

	quantiles = df.stat.approxQuantile("meter_reading", [0.02, 0.98], 0)

	if len(quantiles) > 0:
		return df.where(F.expr("meter_reading > {0} and meter_reading < {1}".format(quantiles[0], quantiles[1])))
	else:
		return spark.createDataFrame((),df.schema)

def get_building(df, building_id):

	return df.where(F.expr("building_id = {0}".format(building_id)))

def get_season(df, months):

	return df.where(F.expr("month in {0}".format(months)))
	
def fit(season):

	cols = ["air_temperature_est", "dew_temperature_est", "wind_speed_est", "meter", "month", "day", "hour"]
	imputer = Imputer(strategy="median", inputCols=["air_temperature", "dew_temperature", "wind_speed"], outputCols=["air_temperature_est", "dew_temperature_est", "wind_speed_est"])
	vector = VectorAssembler(inputCols=cols, outputCol="vector", handleInvalid="error")
	poly = PolynomialExpansion(degree=4, inputCol="vector", outputCol="features")
	regression = LinearRegression(featuresCol="features", labelCol="meter_reading", predictionCol="prediction")
	pipeline = Pipeline(stages=[imputer, vector, poly, regression])
	evaluator = RegressionEvaluator(labelCol="meter_reading", predictionCol="prediction", metricName="rmse")

	params = ParamGridBuilder() \
				.addGrid(imputer.strategy, ["mean", "median"]) \
				.addGrid(poly.degree, [3]) \
				.addGrid(regression.fitIntercept, [True, False]) \
				.addGrid(regression.maxIter, [100]) \
				.addGrid(regression.standardization, [True, False]) \
				.build()

	validator = CrossValidator(estimator=pipeline, estimatorParamMaps=params, evaluator=evaluator, numFolds=3, seed=51)
	model = validator.fit(season)

	return model.bestModel

def predict(model, test, season):
	
	filtered = get_season(test, season)
	predictions = model.transform(filtered)
	
	evaluator = RegressionEvaluator(labelCol="meter_reading", predictionCol="prediction", metricName="rmse")
	rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
	r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})
	mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})

	return (predictions, (season, rmse, r2, mae))

def save_model(building_id, model, season):

	model_path = "output/linear_model_{0}_{1}".format(building_id, season)
	model.write().overwrite().save(model_path)

def train(df, months):

	season = get_season(df, months)
	season = remove_outliers(season)
	model = fit(season)

	return model

print("Loading all data")
df = spark.table("training")

print("Dropping tables")
spark.sql("drop table if exists linear_predictions")
spark.sql("drop table if exists linear_predictions_metrics")

print("Caching splits")
training, test = df.randomSplit([0.8, 0.2])
training.cache()
test.cache()

schema = StructType([StructField("building_id", IntegerType(), False), 
			StructField("season", StringType(), False), 
			StructField("rmse", DoubleType(), False),
			StructField("r2", DoubleType(), False),
			StructField("mae", DoubleType(), False)])

season_months = {"Winter": "(12,1,2)", "Spring": "(3,4,5)", "Summer": "(6,7,8)", "Fall": "(9,10,11)"}

buildings = spark.read.load("../datasets/building_metadata.csv", format="csv", sep=",", inferSchema="true", header="true").select("building_id")

for row in buildings.toLocalIterator():
	
	building_id = row.building_id

	for season, months in season_months.items():
		
		print("Filtering training data for building: {0}".format(building_id))
		building_df = get_building(training, building_id)

		print("Applying fit for season: {0}".format(season))
		model = train(building_df, months)

		print("Saving model")
		save_model(building_id, model, season)
		
		print("Filtering test data for building: {0}".format(building_id))
		building_df = get_building(test, building_id)

		print("Predicting test data")
		prediction = predict(model, building_df, months)

		print("Saving predictions")
		prediction, metrics = prediction
		prediction.coalesce(1).write.saveAsTable("linear_predictions", format="parquet", mode="append")
		
		print("Saving metrics")
		season, rmse, r2, mae = metrics
		metric = spark.createDataFrame([(building_id, season, rmse, r2, mae)], schema)
		metric.coalesce(1).write.saveAsTable("linear_predictions_metrics", format="parquet", mode="append")
