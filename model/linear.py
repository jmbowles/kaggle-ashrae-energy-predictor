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

def load_buildings(building_id=None):

	df = spark.read.load("../datasets/building_metadata.csv", format="csv", sep=",", inferSchema="true", header="true").select("building_id").orderBy("building_id")

	if building_id:
		return df.where(df.building_id >= building_id).orderBy("building_id")
	else:
		return df

def create_series(season, building_id):

	season = season.withColumn("date", F.explode(season.dates))
	season = season.withColumn("hours", F.sequence(F.lit(0), F.lit(23)))
	season = season.withColumn("exploded_hour", F.explode(season.hours))
	season = season.withColumn("building_id", F.lit(building_id))
	season = season.withColumn("meter", F.lit(0))
	season = season.withColumn("timestamp", F.to_timestamp(F.concat(season.date, F.lit(" "), season.exploded_hour, F.lit(":00:00"))))
	season = season.withColumn("month", F.month(season.timestamp))
	season = season.withColumn("day", F.dayofmonth(season.timestamp))
	season = season.withColumn("hour", F.hour(season.timestamp))
	season = season.withColumn("meter_reading", F.lit(1.0))
	season = season.drop("dates", "date", "hours", "exploded_hour")

	return season


def make_season(year, building_id, months):

	from ast import literal_eval

	print("Making season for building {0}, months: {1}".format(building_id, months))
	months_tuple = literal_eval(months)
	start_month = min(months_tuple)
	end_month = max(months_tuple)

	season = None

	if start_month == 1 and end_month == 12:

		sql = "SELECT sequence(to_date('{0}-{1}-01'), to_date('{0}-{2}-25'), interval 1 day) as dates".format(year, 1, 2)
		jan_feb = spark.sql(sql)
		jan_feb = create_series(jan_feb, building_id)

		sql = "SELECT sequence(to_date('{0}-{1}-01'), to_date('{0}-{2}-25'), interval 1 day) as dates".format(year, 12, 12)
		december = spark.sql(sql)
		december = create_series(december, building_id)
		season = jan_feb.union(december)
	else:
		sql = "SELECT sequence(to_date('{0}-{1}-01'), to_date('{0}-{2}-25'), interval 1 day) as dates".format(year, start_month, end_month)
		season = spark.sql(sql)
		season = create_series(season, building_id)

	# Dataframe must be saved prior to joining to prevent implicit cartesian join exception, even though that's not the case
	print("Saving missing data for building {0}".format(building_id))
	season.coalesce(1).write.saveAsTable("missing_data", format="parquet", mode="append")
	season = spark.table("missing_data")

	meta = spark.read.load("../datasets/building_metadata.csv", format="csv", sep=",", inferSchema="true", header="true")
	meta = meta.where(F.expr("building_id = {0}".format(building_id)))
	meta = meta.withColumnRenamed("building_id", "building_id_meta")
	meta = meta.withColumnRenamed("site_id", "site_id_meta")

	weather = spark.read.load("../datasets/weather_train.csv", format="csv", sep=",", inferSchema="true", header="true")
	weather = weather.dropDuplicates(["site_id", "timestamp"])
	weather = weather.withColumnRenamed("timestamp", "timestamp_wx")
	weather = weather.withColumnRenamed("site_id", "site_id_wx")

	print("Joining data for building {0}".format(building_id))
	season = season.join(meta, [meta.building_id_meta == season.building_id])
	season = season.join(weather, [season.timestamp == weather.timestamp_wx, season.site_id_meta == weather.site_id_wx], "left_outer")
	season = season.withColumnRenamed("site_id_meta", "site_id")
	season = season.drop("building_id_meta", "site_id_wx", "timestamp_wx")

	return season


def get_season(df, building_id, months):

	season = df.where(F.expr("month in {0}".format(months)))
	
	if season.count() == 0:
		return make_season(2016, building_id, months)
	else:
		return season
	
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

def predict(model, test, building_id, months):
	
	filtered = get_season(test, building_id, months)
	predictions = model.transform(filtered)
	
	evaluator = RegressionEvaluator(labelCol="meter_reading", predictionCol="prediction", metricName="rmse")
	rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
	r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})
	mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})

	return (predictions, (months, rmse, r2, mae))

def save_model(building_id, model, season):

	model_path = "output/linear_model_{0}_{1}".format(building_id, season)
	model.write().overwrite().save(model_path)

def train(df, building_id, months):

	season = get_season(df, building_id, months)
	#season = remove_outliers(season)
	model = fit(season)

	return model

def save_existing_predictions(building_id):

	temp = spark.sql("select * from linear_predictions where building_id < {0}".format(building_id))
	temp.coalesce(1).write.saveAsTable("linear_predictions_temp", format="parquet", mode="append")
	spark.sql("drop table linear_predictions")
	spark.sql("alter table linear_predictions_temp rename to linear_predictions")


print("Loading all data")
df = spark.table("training")
df = df.withColumn("meter_reading", F.when(df.meter_reading == 0, F.lit(1.0)).otherwise(df.meter_reading))

#print("Dropping tables")
#spark.sql("drop table if exists linear_predictions")
#spark.sql("drop table if exists linear_predictions_metrics")

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

starting_building = 420
save_existing_predictions(starting_building)
buildings = load_buildings(starting_building)

for row in buildings.toLocalIterator():
	
	building_id = row.building_id

	for season, months in season_months.items():
		
		print("Filtering training data for building: {0}".format(building_id))
		training_building = get_building(training, building_id)

		print("Applying fit for season: {0}".format(season))
		model = train(training_building, building_id, months)

		print("Saving model")
		save_model(building_id, model, season)
		
		print("Filtering test data for building: {0}".format(building_id))
		test_building = get_building(test, building_id)

		print("Predicting test data")
		prediction = predict(model, test_building, building_id, months)

		print("Saving predictions")
		prediction, metrics = prediction
		prediction.coalesce(1).write.saveAsTable("linear_predictions", format="parquet", mode="append")
		
		print("Saving metrics")
		season, rmse, r2, mae = metrics
		metric = spark.createDataFrame([(building_id, season, rmse, r2, mae)], schema)
		metric.coalesce(1).write.saveAsTable("linear_predictions_metrics", format="parquet", mode="append")

'''
building_id = 420

for season, months in season_months.items():
	
	print("Filtering training data for building: {0}".format(building_id))
	training_building = get_building(training, building_id)

	print("Applying fit for season: {0}".format(season))
	model = train(training_building, building_id, months)

	print("Saving model")
	save_model(building_id, model, season)
	
	print("Filtering test data for building: {0}".format(building_id))
	test_building = get_building(test, building_id)

	print("Predicting test data")
	prediction = predict(model, test_building, building_id, months)

	print("Saving predictions")
	prediction, metrics = prediction
	prediction.coalesce(1).write.saveAsTable("linear_predictions", format="parquet", mode="append")
	
	print("Saving metrics")
	season, rmse, r2, mae = metrics
	metric = spark.createDataFrame([(building_id, season, rmse, r2, mae)], schema)
	metric.coalesce(1).write.saveAsTable("linear_predictions_metrics", format="parquet", mode="append")
'''

	

