# Databricks notebook source
import json
import os
import urllib.parse
from pyspark.sql.functions import desc, asc, monotonically_increasing_id, collect_list, col, dense_rank, row_number, lit,  floor, concat_ws, udf, struct, sort_array, round

import pyspark.sql.functions as psf
from pyspark.sql.types import StructField, StringType, IntegerType, LongType, StructType, DateType, FloatType, DoubleType
from pyspark.sql.window import Window
import shutil

# COMMAND ----------

def get_unclassified_images(input_flickr_df, input_labelled_images_ids_df):
  # Create "UNCLASSIFIED LABEL" for all images that not have been classified
  return input_flickr_df.select("image_id") \
                        .join(input_labelled_images_ids_df, 'image_id', 'left_anti') \
                        .withColumn('label',lit('unclassified')) \
                        .withColumn('confidence',lit(100.0))

def get_at_least_one_label_per_img(input_df, min_confidence):
  img_id_window = Window.partitionBy(col('image_id'))
  # get for each image the label with the hightest confidence
  top_values_per_label_df = input_df.select("image_id", "confidence").groupBy("image_id").agg(psf.max("confidence").alias("confidence"))
  top_labels_df = input_df.join(top_values_per_label_df, ["image_id", "confidence"]) # will result into multiple, if when ties exist
  top_labels_df = input_df.withColumn('max_confidence', psf.max('confidence').over(img_id_window)).where(col('confidence') == col('max_confidence')).drop("max_confidence")
  # only use labels with confidence larger than MINIMUM_CONFIDENCE
  labels_with_min_confidence_df = input_df.where(col("confidence") > min_confidence)
  # create a union (from the min confidence and the top -> distinct to reduce if top val exist twice due to union)
  return labels_with_min_confidence_df.union(top_labels_df).distinct() 

# COMMAND ----------

labels_df = spark.read.parquet("/mnt/group07/final_data_product/classification_result/labels.parquet")
downloaded_df = spark.read.parquet("/mnt/group07/final_data_product/classification_result/downloaded.parquet") # CURRENTLY NOT LIMIT .limit(50000)
flickr_metadata_df = spark.read.parquet("/mnt/group07/final_data_product/visualization_input.parquet").withColumnRenamed("id", "image_id")

# COMMAND ----------

def get_download_url(row):
  flickr_ur = "https://farm%s.staticflickr.com/%s/%s_%s.%s"
  return flickr_ur % (row[5], row[6], row[0], row[7], row[8])

# COMMAND ----------

unclassified_images_df = spark.read.parquet('/mnt/group07/final_data_product/classification_result/unclassified_images.parquet').cache()
unclassified_images_df.printSchema()

download_link_udf = spark.udf.register("get_download_url", get_download_url, StringType())
download_links_unclassified = unclassified_images_df.select(unclassified_images_df.image_id, download_link_udf(struct([unclassified_images_df[x] for x in unclassified_images_df.columns])).alias("download_link"))

downloaded_images = spark.read.parquet('/mnt/group07/final_data_product/downloaded_images-*.parquet')
downloaded_images.printSchema()

image_bytes_not_classified_df = download_links_unclassified.join(downloaded_images, download_links_unclassified.image_id == downloaded_images.id, 'left').select('image_id', 'img_binary', 'download_link')
image_bytes_not_classified_df.write.parquet('/mnt/group07/final_data_product/classification_result/unclassified_image_bytes.parquet')



# COMMAND ----------

image_bytes_not_classified_df = spark.read.parquet('/mnt/group07/final_data_product/classification_result/unclassified_image_bytes.parquet')
found = image_bytes_not_classified_df.where(image_bytes_not_classified_df.image_id == 91246886).select("image_id", "download_link")
found.show(20, False)