# Databricks notebook source
OUTPUT_DIR = "/mnt/group07/classification_test_files/classification_test"

# COMMAND ----------

tags_df = spark.read.parquet("test_sorted_classification.parquet")
flickr_df = spark.read.parquet("/mnt/group07/full/flickr.parquet")

# COMMAND ----------

tags_df = tags_df.orderBy(["image_id"])

# COMMAND ----------

from pyspark.sql.functions import desc, asc
 
tags_wmeta_df = tags_df.join(flickr_df.select("id", "photo_video_page_url", "title"), tags_df.image_id == flickr_df.id).drop("id")
tags_wmeta_df = tags_wmeta_df.orderBy([asc("label"), desc("confidence"), asc("image_id")])

# COMMAND ----------

tags_wmeta_df.write.mode("overwrite").partitionBy('label').json(os.path.join(OUTPUT_DIR, "tmp", "tags"))
# continue with scala - i suppose

# COMMAND ----------

import os
#tags_wmeta_df.write.mode("overwrite").option("maxRecordsPerFile", 100).partitionBy('label').json(os.path.join(OUTPUT_DIR, "subset"))

# COMMAND ----------

from  pyspark.sql.functions import input_file_name

#df_chucks= spark.read.format("json").load("/mnt/group07/classification_test_files/classification_test")
#df_chucks = df_chucks.withColumn("C_Location",input_file_name())

# COMMAND ----------

#df_chucks.show(truncate=False)

# COMMAND ----------

from pyspark.sql import functions as F
# check: create folder for each tag
# check: write (firt approach) one big json for everything
# check: write chunked version
#       -> chunk big json for each tag into serveral small ones
# TODO: write inv list for all tags that were written (determine name of the chunks? ; )
# --> create df which maps tag to chunckname -> transform that to a json-baed inv list
 # TODO: to satify the app interface: write index.json

# COMMAND ----------

