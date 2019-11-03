from __future__ import print_function
"""

"""
import pyspark.sql.functions as F
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("TransformFeatures") \
	.config("spark.dynamicAllocation.enabled", "true") \
	.config("spark.shuffle.service.enabled", "true") \
	.config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
	.enableHiveSupport() \
	.getOrCreate()

def split_timestamp(df):

	df = df.withColumn("month", F.month(df.timestamp))
	df = df.withColumn("day", F.dayofmonth(df.timestamp))
	df = df.withColumn("hour", F.hour(df.timestamp))

	return df

def create_table(df, table_name):
	print("Dropping table '{0}'".format(table_name))
	spark.sql("drop table if exists {0}".format(table_name))

	print("Saving table '{0}'".format(table_name))
	df.coalesce(4).write.saveAsTable(table_name, format="parquet", mode="overwrite")

print("Loading and caching data")
train = spark.read.load("../datasets/train.csv", format="csv", sep=",", inferSchema="true", header="true")
train = train.dropDuplicates(["building_id", "meter", "timestamp"])
train.cache()
print("Training dataset row count: {0}".format(train.count()))

test = spark.read.load("../datasets/test.csv", format="csv", sep=",", inferSchema="true", header="true")
test = test.dropDuplicates(["building_id", "meter", "timestamp"])
test.cache()
print("Test dataset row count: {0}".format(test.count()))

meta = spark.read.load("../datasets/building_metadata.csv", format="csv", sep=",", inferSchema="true", header="true")
meta = meta.withColumnRenamed("building_id", "building_id_meta")
meta = meta.withColumnRenamed("site_id", "site_id_meta")
print("Metadata row count: {0}".format(meta.count()))

weather = spark.read.load("../datasets/weather_train.csv", format="csv", sep=",", inferSchema="true", header="true")
weather = weather.dropDuplicates(["site_id", "timestamp"])
weather = weather.withColumnRenamed("timestamp", "timestamp_wx")
weather = weather.withColumnRenamed("site_id", "site_id_wx")
print("Weather row count: {0}".format(weather.count()))

#weather.withColumn("month_wx", F.month(weather.timestamp_wx)).withColumn("day_wx", F.dayofmonth(weather.timestamp_wx)).groupby("site_id_wx", "month_wx", "day_wx").agg(F.avg(weather.air_temperature).alias("avg_celsius")).withColumn("avg_fahrenheit", F.expr("avg_celsius * 1.8 + 32")).show()
#weather.where(F.expr("site_id_wx = 2 and month(timestamp_wx) = 1")).withColumn("month_wx", F.month(weather.timestamp_wx)).withColumn("day_wx", F.dayofmonth(weather.timestamp_wx)).groupby("site_id_wx", "month_wx", "day_wx").agg(F.avg(weather.air_temperature).alias("avg_celsius")).withColumn("avg_fahrenheit", F.expr("avg_celsius * 1.8 + 32")).orderBy("day_wx").show(50)

train = train.join(meta, [meta.building_id_meta == train.building_id])
train = train.join(weather, [train.timestamp == weather.timestamp_wx, train.site_id_meta == weather.site_id_wx], "left_outer")
train = train.withColumnRenamed("site_id_meta", "site_id")
train = train.drop("building_id_meta", "site_id_wx", "timestamp_wx")
print("Training joined row count: {0}".format(train.count()))

test = test.join(meta, [meta.building_id_meta == test.building_id])
test = test.join(weather, [test.timestamp == weather.timestamp_wx, test.site_id_meta == weather.site_id_wx], "left_outer")
test = test.withColumnRenamed("site_id_meta", "site_id")
test = test.drop("building_id_meta", "site_id_wx", "timestamp_wx")
print("Test joined row count: {0}".format(test.count()))

print("Transforming datasets")
train = split_timestamp(train)
test = split_timestamp(test)

print("Creating tables")
create_table(train, "training")
create_table(test, "test")







