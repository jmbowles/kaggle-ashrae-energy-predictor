from __future__ import print_function
"""


"""
from pyspark.sql.types import DoubleType
from pyspark.ml import PipelineModel

import pyspark.sql.functions as F
from pyspark.sql import SparkSession

pipeline_model_path = "output/logistic_pipeline_model"
csv_path = "output/submit/submittal_8.csv"

spark = SparkSession.builder.appName("LogisticSubmittal") \
	.config("spark.dynamicAllocation.enabled", "true") \
	.config("spark.shuffle.service.enabled", "true") \
	.config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
	.getOrCreate()

print("Loading Data")
df = spark.read.load("../datasets/test.csv", format="csv", sep=",", inferSchema="true", header="true")
df.cache()

continuous_cols = ["Census_OSBuildNumber", "AVProductStatesIdentifier", "AVProductsInstalled", "OsBuild", "OsSuite", "IeVerIdentifier", 
					"Census_ProcessorCoreCount", "Census_TotalPhysicalRAM", "Census_InternalBatteryNumberOfCharges", "Census_OSBuildRevision"]
fill_0_cols = ["OrganizationIdentifier", "RtpStateBitfield", "Wdft_RegionIdentifier", "Census_ProcessorManufacturerIdentifier", "Census_IsFlightsDisabled", "Census_IsFlightingInternal", 
				"HasTpm", "Census_OSUILocaleIdentifier", "Census_IsVirtualDevice", "SMode", "IsBeta", "Wdft_IsGamer", "Census_IsAlwaysOnAlwaysConnectedCapable", 
				"Census_IsPortableOperatingSystem", "Census_OEMModelIdentifier", "Census_OEMNameIdentifier", "IsSxsPassiveMode", "Census_ThresholdOptIn", "CountryIdentifier", 
				"Census_IsWIMBootEnabled", "Census_OSInstallLanguageIdentifier", "Census_IsPenCapable", "Census_ProcessorModelIdentifier", "Census_IsSecureBootEnabled", "AutoSampleOptIn",
				"UacLuaenable", "Census_IsTouchEnabled", "Census_HasOpticalDiskDrive", "Firewall", "IsProtected", "Census_InternalPrimaryDiagonalDisplaySizeInInches", "AVProductsEnabled", "GeoNameIdentifier"]
fill_unknown_cols = ["Census_MDC2FormFactor", "OsBuildLab", "Platform", "Census_OSArchitecture", "Census_DeviceFamily", "Census_PowerPlatformRoleName", "Census_OSSkuName", 
						"SkuEdition", "Census_ActivationChannel", "OsPlatformSubRelease", "Census_ChassisTypeName", "Census_FlightRing", "Census_OSInstallTypeName", 
						"Processor", "Census_OSWUAutoUpdateOptionsName", "ProductName", "Census_OSEdition", "Census_GenuineStateName", "Census_PrimaryDiskTypeName", "PuaMode"]
 
df = df.replace("requireAdmin", "RequireAdmin", ["SmartScreen"])
df = df.replace("on", "1", ["SmartScreen"])
df = df.replace("On", "1", ["SmartScreen"])
df = df.replace("Enabled", "1", ["SmartScreen"])
df = df.replace("prompt", "Prompt", ["SmartScreen"])
df = df.replace("Promt", "Prompt", ["SmartScreen"])
df = df.replace("00000000", "0", ["SmartScreen"])
df = df.replace("off", "0", ["SmartScreen"])
df = df.replace("OFF", "0", ["SmartScreen"])
df = df.replace("warn", "Warn", ["SmartScreen"])
df = df.fillna("0", ["SmartScreen"])
df = df.fillna(0, continuous_cols + fill_0_cols)
df = df.fillna("UNKNOWN", fill_unknown_cols)
df = df.fillna(9999999, ["Census_PrimaryDiskTotalCapacity"])
df = df.fillna(1000000, ["Census_InternalPrimaryDisplayResolutionHorizontal", "Census_InternalPrimaryDisplayResolutionVertical"])
df = df.withColumn("Derived_AvSigVersion", F.regexp_replace(df.AvSigVersion, r"[^0-9]", "").cast("integer"))
df = df.withColumn("Derived_AppVersion", F.regexp_replace(df.AppVersion, r"[^0-9]", "").cast("integer"))
df = df.withColumn("Derived_EngineVersion", F.regexp_replace(df.EngineVersion, r"[^0-9]", "").cast("integer"))
df = df.withColumn("Derived_OsVer", F.regexp_replace(df.OsVer, r"[^0-9]", "").cast("integer"))
df = df.withColumn("Derived_CensusOSVersion", F.regexp_replace(df.Census_OSVersion, r"[^0-9]", "").cast("integer"))
df = df.withColumn("Derived_Firmware", df.Census_FirmwareManufacturerIdentifier + df.Census_FirmwareVersionIdentifier)
df = df.withColumn("Derived_VolumeCapacity", F.round(df.Census_SystemVolumeTotalCapacity / df.Census_PrimaryDiskTotalCapacity, 2))
df = df.withColumn("Derived_Resolution", df.Census_InternalPrimaryDisplayResolutionHorizontal * df.Census_InternalPrimaryDisplayResolutionVertical)

derived_cols = [col for col in df.columns if col.startswith("Derived_")]

df = df.fillna(0, derived_cols)

print("Loading Pipeline")
pipeline_model = PipelineModel.load(pipeline_model_path)

print("Fitting for Submittal")
predictions = pipeline_model.transform(df)
predictions.select("MachineIdentifier", "probability", "prediction").show(truncate=False)

print("Creating CSV for Submittal")

# Silly workaround for extracting an element from a dense or sparse vector. Probability column is a vector, with probs for each label
# https://stackoverflow.com/questions/39555864/how-to-access-element-of-a-vectorudt-column-in-a-spark-dataframe
def vector_item_(vector_column, index):
    try:
        return float(vector_column[index])
    except ValueError:
        return None

vector_item = F.udf(vector_item_, DoubleType())

df_submit = predictions.withColumn("Label_0", vector_item("probability", F.lit(0)))
df_submit = df_submit.withColumn("Label_1", vector_item("probability", F.lit(1)))
df_submit = df_submit.withColumn("HasDetections", df_submit.Label_1)
df_submit = df_submit.select("MachineIdentifier", "HasDetections")

# Yet another workaround to write to a CSV file
df_submit.coalesce(1).toPandas().to_csv(csv_path, header=True, index=False)

print("Total rows written to file: {0}".format(df_submit.count()))