# Databricks notebook source
# DBTITLE 1,FOR CPU Classified
from pyspark.sql.functions import lit
BUCKET_NUM=7

classified_ids_df = spark.read.parquet("/mnt/group07/final_data_product/classified_images/{0}-cpu/images_classified.parquet/".format(BUCKET_NUM)).select("id").distinct()
downloaded_files_df = spark.read.parquet("/mnt/group07/final_data_product/downloaded_images-{0}.parquet".format(BUCKET_NUM))
print(downloaded_files_df.count())
keep_binary_data = downloaded_files_df.join(classified_ids_df, "id", "leftanti")
drop_binary_data = downloaded_files_df.drop("img_binary").join(classified_ids_df, "id", "inner").withColumn("img_binary", lit(None))
keep_binary_data.unionByName(drop_binary_data).write.mode('overwrite').format("parquet").save("/mnt/group07/final_data_product/downloaded_images-{0}.parquet".format(BUCKET_NUM))
downloaded_files_df = spark.read.parquet("/mnt/group07/final_data_product/downloaded_images-{0}.parquet".format(BUCKET_NUM))
print(downloaded_files_df.count())

# COMMAND ----------

files = dbutils.fs.ls("/mnt/group07/final_data_product/downloaded_images-{0}.parquet".format(BUCKET_NUM))
bytes_size = 0
for file in files:
  bytes_size += file.size
print("/mnt/group07/final_data_product/downloaded_images-{0}.parquet".format(BUCKET_NUM))
print(bytes_size)

# COMMAND ----------

