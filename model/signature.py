from __future__ import print_function
"""

"""
import numpy as np
import pandas as pd

import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.window import Window


spark = SparkSession.builder \
	.appName("AvSignature TimeSeries") \
	.config("spark.sql.execution.arrow.enabled", "true") \
	.config("spark.dynamicAllocation.enabled", "true") \
	.config("spark.shuffle.service.enabled", "true") \
	.enableHiveSupport() \
	.getOrCreate()


print("Loading and Caching Data")
df = spark.read.table("training")
df.cache()

places = Window.partitionBy("AvSigVersion", "Census_OSVersion", "Wdft_RegionIdentifier", "CountryIdentifier", "AVProductStatesIdentifier", "SmartScreen")
detections = Window.partitionBy("HasDetections", "AvSigVersion", "Census_OSVersion", "Wdft_RegionIdentifier", "CountryIdentifier", "AVProductStatesIdentifier", "SmartScreen")
total_detections = F.count("HasDetections").over(places)
detection = F.count("HasDetections").over(detections)

df1 = df.withColumn("elapsed_days", F.datediff(df.OSVersionUTC, df.AvSigVersionUTC)) \
		.withColumn("total_detections", total_detections) \
		.withColumn("count_0", F.when(F.expr("HasDetections == 0"), detection).otherwise(total_detections - detection)) \
		.withColumn("count_1", total_detections - F.col("count_0")) \
		.withColumn("ratio_0", F.round(F.expr("count_0 / total_detections"), 2)) \
		.withColumn("ratio_1", F.round(F.expr("count_1 / total_detections"), 2))
#df1 = df1.select("HasDetections", "AvSigVersion", "AvSigVersionUTC", "Wdft_RegionIdentifier", "CountryIdentifier", "total_detections", "count_0", "count_1").orderBy("AvSigVersionUTC", "Wdft_RegionIdentifier", "CountryIdentifier", "HasDetections")
#df1.show()

selected_cols = ["AvSigVersion", "Census_OSVersion", "AvSigVersionUTC", "OSVersionUTC", "Wdft_RegionIdentifier", "CountryIdentifier", "AVProductStatesIdentifier", "SmartScreen", "elapsed_days", "total_detections", "count_0", "count_1", "ratio_0", "ratio_1"]
df1 = df1.select(*selected_cols)
#df1 = df1.withColumn("AvSigVersionUTC_Hour", F.date_trunc("hour", df1.AvSigVersionUTC))
#df1 = df1.withColumn("OSVersionUTC_Hour", F.date_trunc("hour", df1.OSVersionUTC))
#df1 = df1.distinct().orderBy("AvSigVersionUTC_Hour", "OSVersionUTC_Hour", "Wdft_RegionIdentifier").select(*selected_cols)
#df1.where(F.expr("total_detections >= 50 and Wdft_RegionIdentifier = 7 and CountryIdentifier = 43 and AVProductStatesIdentifier = 53447")).distinct().orderBy("AvSigVersionUTC_Hour", "OSVersionUTC_Hour").select(*selected_cols).show(1000)