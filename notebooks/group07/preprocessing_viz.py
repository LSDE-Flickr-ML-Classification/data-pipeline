# Databricks notebook source
# MAGIC %run /group07/shared

# COMMAND ----------

import os
import pyspark.sql.functions as f

# COMMAND ----------

OUTPUT_DIR = "/mnt/group07/final_data_product"
# Load FULL CSV Dataset
df_flickr = spark.read.format("CSV").option("delimiter", "\t").schema(get_csv_data_scheme()).load("/mnt/data/flickr/yfcc100m_dataset-*.bz2")

# COMMAND ----------

df_flickr.printSchema()

# COMMAND ----------

# filter on images only
df_flickr_images_only = df_flickr.where(df_flickr.photo_video_marker == 0)

# COMMAND ----------

# select all relevant fields + repartition across cluster TODO
# sort for even partitions
df_flickr_images_only = df_flickr_images_only.sort("id")

# COMMAND ----------

df_visualization = df_flickr_images_only.select("id", "title", "user_nsid", "photo_video_farm_id", "photo_video_server_id", "photo_video_secret", "photo_video_extension_original")

# COMMAND ----------

# Write visualization input (and ensure sorting)
df_visualization.sort("id").write.parquet(os.path.join(OUTPUT_DIR, "visualization_input.parquet"))

# COMMAND ----------

df_visualization.count()