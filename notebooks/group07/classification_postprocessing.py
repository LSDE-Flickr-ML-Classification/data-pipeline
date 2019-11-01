# Databricks notebook source
OUTPUT_DIR = "/mnt/group07/final_data_product/classification_result/"

# COMMAND ----------

from pyspark.sql.functions import col, explode
import os

# COMMAND ----------

def read_imgs_gpu_classified(bucket_num):
  df = spark.read.option("header","true").csv("/mnt/group07/final_data_product/classified_images/{0}/part*.csv".format(bucket_num)).withColumnRenamed("id", "image_id")
  df = df.withColumn("confidence", col("confidence").cast("double"))
  return df

def read_downloads_stored(bucket_num):
  df = spark.read.parquet("/mnt/group07/final_data_product/downloaded_images-{0}.parquet".format(bucket_num)).select("id", "img_size", "img_width", "img_height", "error_code")
  return df

def filter_for_sucessful_downloads(df):
  df = df.where( (col("error_code") == "200") | (col("error_code") == "s3" ) ).select( col("id").alias("image_id") )
  return df

def read_imgs_cpu_classified(bucket_num):
  df = spark.read.parquet("/mnt/group07/final_data_product/classified_images/{0}-cpu/images_classified.parquet".format(bucket_num)).select(col("id").alias("image_id"), "classes")
  df = df.withColumn("label_struct", explode(col("classes"))).drop("classes").select("image_id", "label_struct.*")
  return df

def read_download_and_classification_imgs_classified(bucket_num):
  df = spark.read.parquet("/mnt/group07/final_data_product/classified_images/{0}-cpu/images_classified.parquet".format(bucket_num)).select(col("id").alias("image_id"), "classes")
  df = df.withColumn("label_struct", explode(col("classes"))).drop("classes").select("image_id", "label_struct.*")
  return df

def read_download_and_classification_imgs_downloaded(bucket_num):
  df = spark.read.parquet("/mnt/group07/final_data_product/classified_images/{0}-cpu/images_classified.parquet".format(bucket_num)).select("id", "img_size", "img_width", "img_height", "error_code")
  return df

def filter_download_and_classification_imgs_downloaded(df):
  df = df.where( (col("error_code") == "200") | (col("error_code") == "s3" ) ).select( col("id").alias("image_id"))
  return df

# COMMAND ----------

# DBTITLE 1,Read GPU Classified Images + Gather All Downloaded Images from Bucket 0
# Read both the classified images + the flickr dataset
b0_df = read_imgs_gpu_classified(0)
b0_downloads_df = read_downloads_stored(0)
b0_downloads_df.groupBy("error_code").count().show()
b0_downloads_stats_df = b0_downloads_df
b0_downloads_df = filter_for_sucessful_downloads(b0_downloads_df)

# COMMAND ----------

# DBTITLE 1,Read GPU Classified Images + Gather All Downloaded Images from Bucket 1
# Read both the classified images + the flickr dataset
b1_df = read_imgs_gpu_classified(1)
b1_downloads_df = read_downloads_stored(1)
b1_downloads_df.groupBy("error_code").count().show()
b1_downloads_stats_df = b1_downloads_df
b1_downloads_df = filter_for_sucessful_downloads(b1_downloads_df)

# COMMAND ----------

# DBTITLE 1,Read GPU Classified Images + Gather All Downloaded Images from Bucket 2
# Read both the classified images + the flickr dataset
b2_df = read_imgs_gpu_classified(2)
b2_downloads_df = read_downloads_stored(2)
b2_downloads_df.groupBy("error_code").count().show()
b2_downloads_stats_df = b2_downloads_df
b2_downloads_df = filter_for_sucessful_downloads(b2_downloads_df)

# COMMAND ----------

# DBTITLE 1,Read GPU Classified Images + Gather All Downloaded Images from Bucket 3
# Read both the classified images + the flickr dataset
b3_df = read_imgs_gpu_classified(3)
b3_downloads_df = read_downloads_stored(3)
b3_downloads_df.groupBy("error_code").count().show()
b3_downloads_stats_df = b3_downloads_df
b3_downloads_df = filter_for_sucessful_downloads(b3_downloads_df)

# COMMAND ----------

# Read both the classified images + the flickr dataset
b4_df = read_imgs_gpu_classified(4)
b4_downloads_df = read_downloads_stored(4)
b4_downloads_df.groupBy("error_code").count().show()
b4_downloads_stats_df = b4_downloads_df
b4_downloads_df = filter_for_sucessful_downloads(b4_downloads_df)

# COMMAND ----------

# Read both the classified images + the flickr dataset
b5_df = read_imgs_gpu_classified(5)
b5_downloads_df = read_downloads_stored(5)
b5_downloads_df.groupBy("error_code").count().show()
b5_downloads_stats_df = b5_downloads_df
b5_downloads_df = filter_for_sucessful_downloads(b5_downloads_df)

# COMMAND ----------

# Read both the classified images + the flickr dataset
b6_df = read_imgs_gpu_classified(6)
b6_downloads_df = read_downloads_stored(6)
b6_downloads_df.groupBy("error_code").count().show()
b6_downloads_stats_df = b6_downloads_df
b6_downloads_df = filter_for_sucessful_downloads(b6_downloads_df)

# COMMAND ----------

# DBTITLE 1,Read CPU Classified Images from Bucket 7
b7_df = read_imgs_cpu_classified(7)
# this bucket was "prestored":
b7_downloads_df = read_downloads_stored(7)
b7_downloads_df.groupBy("error_code").count().show()
b7_downloads_stats_df = b7_downloads_df
b7_downloads_df = filter_for_sucessful_downloads(b7_downloads_df)

# COMMAND ----------

# DBTITLE 1,Read GPU Classified Images from Greyscale
b7_df_greyscale_error = read_imgs_gpu_classified(7)
print(b7_df_greyscale_error.count())
b7_downloads_df_greyscale_error = b7_df_greyscale_error.select("image_id").distinct()

# COMMAND ----------

# DBTITLE 1,Read CPU Classified Images from Bucket 8
b8_df = read_download_and_classification_imgs_classified(8)
b8_downloads_df = read_download_and_classification_imgs_downloaded(8)
b8_downloads_df.groupBy("error_code").count().show()
b8_downloads_stats_df = b8_downloads_df
b8_downloads_df = filter_download_and_classification_imgs_downloaded(b8_downloads_df)

# COMMAND ----------

# DBTITLE 1,Read CPU Classified Images from Bucket 9
b9_df = read_download_and_classification_imgs_classified(9)
b9_downloads_df = read_download_and_classification_imgs_downloaded(9)
b9_downloads_df.groupBy("error_code").count().show()
b9_downloads_stats_df = b9_downloads_df
b9_downloads_df = filter_download_and_classification_imgs_downloaded(b9_downloads_df)

# COMMAND ----------

# DBTITLE 1,Read GPU Classified Images + Gather All Downloaded Images from Failed Downloads
# Read both the classified images + the flickr dataset
# and drop duplicates, due to restart added some images twice
b10_df = read_imgs_gpu_classified(10).dropDuplicates()
b10_downloads_df = df = spark.read.parquet("/mnt/group07/final_data_product/downloaded_images-retry-final_v2.parquet").select("id", "img_size", "img_width", "img_height", "error_code").dropDuplicates()
b10_downloads_df.groupBy("error_code").count().show()
b10_downloads_stats_df = b10_downloads_df
b10_downloads_df = filter_for_sucessful_downloads(b10_downloads_df)

# COMMAND ----------

import pyspark.sql.functions as f
b10_df.groupBy("image_id").count().orderBy(f.desc("count")).show(100, truncate=False)

# COMMAND ----------

b10_df.count()

# COMMAND ----------

# DBTITLE 1,Create Final Label Parque
from functools import reduce
from pyspark.sql import DataFrame
from pyspark.sql.functions import lower, col, round, regexp_replace

dfs = [b0_df, b1_df, b2_df, b3_df, b4_df, b5_df, b6_df, b7_df, b8_df, b9_df, b7_df_greyscale_error, b10_df]
labels_df = reduce(DataFrame.unionByName, dfs) \
                .withColumn("label",lower(col("label"))) \
                .withColumn("label", regexp_replace('label', '_', ' ')) \
                .withColumn("confidence", round(col("confidence"), 2))

labels_df.write.mode("overwrite").format("parquet").save(os.path.join(OUTPUT_DIR, "labels.parquet"))
print(labels_df.count())
labels_df.show()

# COMMAND ----------

# DBTITLE 1,Create Final Downloaded Parque
from functools import reduce
from pyspark.sql import DataFrame

dfs = [b0_downloads_df, b1_downloads_df,b2_downloads_df, b3_downloads_df, b4_downloads_df, b5_downloads_df, b6_downloads_df, b7_downloads_df, b8_downloads_df, b9_downloads_df, b7_downloads_df_greyscale_error, b10_downloads_df]
downloads_df = reduce(DataFrame.unionByName, dfs).distinct()
downloads_df.write.mode("overwrite").format("parquet").save(os.path.join(OUTPUT_DIR, "downloaded.parquet"))
print(downloads_df.count())
downloads_df.show()

# COMMAND ----------

# DBTITLE 1,Create Stats
from functools import reduce
from pyspark.sql import DataFrame

dfs = [b0_downloads_stats_df, b1_downloads_stats_df,b2_downloads_stats_df, b3_downloads_stats_df, b4_downloads_stats_df, b5_downloads_stats_df, b6_downloads_stats_df, b7_downloads_stats_df, b8_downloads_stats_df, b9_downloads_stats_df]
stats_df = reduce(DataFrame.unionByName, dfs)
# Remove all stats that where redownloaded:
stats_df = stats_df.join(b10_downloads_stats_df, "id", "leftanti")
print(stats_df.count())
# Add the information from the redownloaded ones
stats_df = stats_df.union(b10_downloads_stats_df)
print(stats_df.count())
stats_df.write.mode("overwrite").format("parquet").save(os.path.join(OUTPUT_DIR, "stats.parquet"))
print(stats_df.count())
stats_df.show()