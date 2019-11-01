# Databricks notebook source
from pyspark.sql.types import StructField, StringType, IntegerType, LongType, StructType, DateType, FloatType

# DataFrame definitions
def get_csv_data_scheme():
  # CSV file
  return StructType(fields=[
    StructField('id', StringType(), True),
    StructField('user_nsid', StringType(), True),
    StructField('user_nickname', StringType(), True),
    StructField('date_taken', StringType(), True),
    StructField('date_uploaded', StringType(), True),
    StructField('capture_device', StringType(), True),
    StructField('title', StringType(), True),
    StructField('description', StringType(), True),
    StructField('user_tags', StringType(), True),
    StructField('machine_tags', StringType(), True),
    StructField('longitude', StringType(), True),
    StructField('latitude', StringType(), True),
    StructField('accuracy', StringType(), True),
    StructField('photo_video_page_url', StringType(), True),
    StructField('photo_video_download_url', StringType(), True),
    StructField('license_name', StringType(), True),
    StructField('license_url', StringType(), True),
    StructField('photo_video_server_id', StringType(), True),
    StructField('photo_video_farm_id', StringType(), True),
    StructField('photo_video_secret', StringType(), True),
    StructField('photo_video_original_secret', StringType(), True),
    StructField('photo_video_extension_original', StringType(), True),
    StructField('photo_video_marker', StringType(), True)
  ])

# COMMAND ----------

# Download History
def get_download_history_schema():
  return StructType(fields= [
    StructField('id', LongType(), False),
    StructField('status_code', IntegerType(), True),
    StructField('download_duration', FloatType(), True),
    StructField('image_size', LongType(), True),
    StructField('image_px_width', IntegerType(), True),
    StructField('image_px_height', IntegerType(), True),
    StructField('image_path', StringType(), True),
  ])

# COMMAND ----------

def get_col_analysis_schema():
  return StructType(fields=[
    StructField('column_name', StringType(), False),
    StructField('total_value_count', IntegerType(), False),
    StructField('null_value_count', IntegerType(), False)
  ])

# COMMAND ----------

