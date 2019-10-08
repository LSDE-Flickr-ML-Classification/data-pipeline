# Databricks notebook source
OUTPUT_DIR = "/mnt/group07/pipeline"

# COMMAND ----------

tags_df = spark.read.parquet("dummy_test.parquet")
flickr_df = spark.read.parquet("/mnt/group07/full/flickr.parquet")

# COMMAND ----------

tags_df = tags_df.orderBy(["image_id"])

# COMMAND ----------

import os
 
tags_wmeta_df = tags_df.join(flickr_df.select("id", "photo_video_page_url", "title"), tags_df.image_id == flickr_df.id).drop("id")
tags_wmeta_df = tags_wmeta_df.orderBy(["tag", "certainty", "image_id"])

# COMMAND ----------

from pyspark.sql.functions import col

tags_tag_cnt_df = tags_wmeta_df.groupBy("tag").count().orderBy("count", ascending=False)
tags_tag_cnt_df = tags_tag_cnt_df.filter(col("count") > 20)

# COMMAND ----------

from pyspark.sql import functions as F

total = tags_tag_cnt_df.groupBy().agg(F.sum("count"))
total.show()

# COMMAND ----------

tags_subset_df = tags_wmeta_df.join(tags_tag_cnt_df, tags_tag_cnt_df.tag == tags_wmeta_df.tag).drop(tags_tag_cnt_df.tag).drop("count")


# COMMAND ----------

tags_subset_df.write.mode("overwrite").option("maxRecordsPerFile", 100).partitionBy('tag').json(os.path.join(OUTPUT_DIR, "subset"))

# COMMAND ----------

from pyspark.sql import functions as F
# check: create folder for each tag
# check: write (firt approach) one big json for everything
# ceck: write chunked version
#       -> chunk big json for each tag into serveral small ones
# TODO: write inv list for all tags that were written (determine name of the chunks? ; )
# --> create df which maps tag to chunckname -> transform that to a json-baed inv list
 # TODO: to satify the app interface: write index.json

# COMMAND ----------

