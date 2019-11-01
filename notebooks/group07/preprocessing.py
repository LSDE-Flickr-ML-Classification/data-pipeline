# Databricks notebook source
# MAGIC %run /group07/shared

# COMMAND ----------

import os
import pyspark.sql.functions as f

# COMMAND ----------

OUTPUT_DIR = "/mnt/group07/final_data_product"
BUCKET_NUM = 9
# Load FULL CSV Dataset
df_flickr = spark.read.format("CSV").option("delimiter", "\t").schema(get_csv_data_scheme()).load("/mnt/data/flickr/yfcc100m_dataset-{0}.bz2".format(BUCKET_NUM))
# done: 0 1 2 3 4 5 6 7 8 9

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

# Write classification (and ensure sorting!)
df_classification.sort("id").write.parquet(os.path.join(OUTPUT_DIR, "download_input-{0}.parquet".format(BUCKET_NUM)))

# COMMAND ----------

df_classification.count()

# COMMAND ----------

# 0 -
# 1 9975029
# 2 9974918
# 3 9974855
# 4 9974782
# 5 9974667
# 6 9842167
# 7 9838844
# 8 9838158
# 9 9838492