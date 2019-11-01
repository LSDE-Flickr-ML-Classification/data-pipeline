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

# filter on images only
df_flickr_images_only = df_flickr.where(df_flickr.photo_video_marker == 0)

# COMMAND ----------

# select all relevant fields + repartition across cluster TODO
# sort for even partitions
df_flickr_images_only = df_flickr_images_only.sort("id")

# COMMAND ----------

# select only id + download url (download + classification)
df_classification = df_flickr_images_only.select("id", "photo_video_download_url", "longitude", "latitude", "photo_video_extension_original")

# COMMAND ----------

stats_df = spark.read.parquet("/mnt/group07/final_data_product/classification_result/stats.parquet")
stats_filtered = stats_df.where(f.col("error_code").isin(["200", "s3", "404", "410", "500"]) == False)
display(stats_filtered.groupBy("error_code").count())

# COMMAND ----------

stats_filtered.select("id").count()

# COMMAND ----------

df_download_input = df_classification.join(stats_filtered.select("id"), "id")

# COMMAND ----------

df_download_input.count()

# COMMAND ----------

# Write classification (and ensure sorting!)
df_download_input.sort("id").write.parquet(os.path.join(OUTPUT_DIR, "download_input-retry.parquet"))

# COMMAND ----------

df_download_input.count()

# COMMAND ----------

