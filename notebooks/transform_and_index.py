# Databricks notebook source
tags_df = spark.read.parquet("dummy_test.parquet")
flickr_df = spark.read.parquet("/mnt/group07/full/flickr.parquet")

# COMMAND ----------

tags_df.show()

# COMMAND ----------

# TODO: create folder for each tag
# TODO: write (firt approach) one big json for everything
# TODO: write final inverted list over all tags (and only one chunk)
# TODO: write chunked version
#       -> chunk big json for each tag into serveral small ones
#       -> update inverted list so that for each chunk there is one entry
# TODO: to satify the app interface: write index.json

# COMMAND ----------

