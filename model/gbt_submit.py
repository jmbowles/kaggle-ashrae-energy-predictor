from __future__ import print_function
"""
0: electricity, 1: chilledwater, 2: steam, 3: hotwater
"""
from pyspark.ml import PipelineModel

import pyspark.sql.functions as F
from pyspark.sql import SparkSession


spark = SparkSession.builder.appName("Linear Regression Submittal") \
	.config("spark.dynamicAllocation.enabled", "true") \
	.config("spark.shuffle.service.enabled", "true") \
	.enableHiveSupport() \
	.getOrCreate()


def get_building(df, building_id):

	return df.where(F.expr("building_id = {0}".format(building_id)))
	
def load_model(building_id):

	model_path = "output/gbt_model_{0}".format(building_id)
	return PipelineModel.load(model_path)

def to_csv(submit_id):

	import os

	file_name = "submittal_{0}.csv".format(submit_id)
	outdir = "./output/submit"

	if not os.path.exists(outdir):
	    os.mkdir(outdir)

	path = os.path.join(outdir, file_name)

	predictions = spark.table("submitted_predictions")
	predictions.select("row_id", "meter_reading").coalesce(1).toPandas().to_csv(path, header=True, index=False)
	print("Total rows written to '{0}': {1}".format(file_name, predictions.count()))


print("Loading test data for prediction submittal")
test = spark.table("test")
test.cache()

submit_id = 1
algo = "gbt"

buildings = spark.read.load("../datasets/building_metadata.csv", format="csv", sep=",", inferSchema="true", header="true").select("building_id")

for row in buildings.toLocalIterator():
	
	building_id = row.building_id

	print("Predicting meter readings for building {0}".format(building_id))
	building = get_building(test, building_id)
	model = load_model(building_id)
	predictions = model.transform(building)

	print("Saving submission")
	predictions = predictions.withColumn("submitted_ts", F.current_timestamp())
	predictions = predictions.withColumn("submit_id", F.lit(submit_id))
	predictions = predictions.withColumn("algo", F.lit(algo))
	predictions = predictions.withColumnRenamed("prediction", "meter_reading").select("row_id", "building_id", "meter", "timestamp", "meter_reading", "submit_id", "submitted_ts", "algo")
	predictions.coalesce(1).write.saveAsTable("submitted_predictions", format="parquet", mode="append")
				
		
to_csv(submit_id)





