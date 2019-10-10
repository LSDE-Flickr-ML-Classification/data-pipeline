# Databricks notebook source
import json
import os
from pyspark.sql.functions import desc, asc, monotonically_increasing_id, collect_list, col, dense_rank, row_number, lit,  floor, concat_ws, udf, struct
from pyspark.sql.types import StructField, StringType, IntegerType, LongType, StructType, DateType, FloatType
from pyspark.sql.window import Window

OUTPUT_DIR = "/mnt/group07/classification_test_files/classification_test"
LABEL_BUCKET_JSON_SIZE = 100

# COMMAND ----------

# Read both the classified images + the flickr dataset
label_df = spark.read.parquet("/mnt/group07//classification_result.parquet")
flickr_df = spark.read.parquet("/mnt/group07/full/flickr.parquet")

# COMMAND ----------

# Combine the flickr dataset (specific cols only) and the classification result
tags_wmeta_df = label_df.join(flickr_df.select("id", "photo_video_page_url", "title"), label_df.image_id == flickr_df.id).drop("id")
# Sort so confidence for each label is at the top
tags_wmeta_df = tags_wmeta_df.orderBy([asc("label"), desc("confidence"), asc("image_id")])

# COMMAND ----------

# Add a uniq id for the correct ordering
tags_wmeta_df = tags_wmeta_df.withColumn("uniq_id", monotonically_increasing_id())

# COMMAND ----------

def json_transform(row):
  result_dict = {
    "confidence": row.confidence,
    "image_id": "{0}".format(row.image_id),
    "flickr_url": "{0}".format(row.photo_video_page_url),
    "title": "{0}".format(row.title)
  }

  return [row.label, row.uniq_id, json.dumps(result_dict, sort_keys=True)]

# Create the json output for each image and store as df
output_rdd = tags_wmeta_df.rdd.map(json_transform)
output_label_df = output_rdd.toDF(['label', "uniq_id", "output"])

# COMMAND ----------

# create a group number for each group
output_label_grouped_df = output_label_df.withColumn("group_id", dense_rank().over(Window.orderBy(output_label_df.label)))

# number all items within a group with an index from 0 to n
output_label_grouped_ids_df = output_label_grouped_df.withColumn("group_row_num", row_number().over(Window.partitionBy(output_label_grouped_df.label).orderBy(output_label_grouped_df.uniq_id)))
output_label_grouped_ids_df = output_label_grouped_ids_df.withColumn("group_row_num", output_label_grouped_ids_df.group_row_num - lit(1))

# COMMAND ----------

# Determine the amount of buckets by dividing the group internal numbering by the bucket size
output_label_grouped_ids_parts_df = output_label_grouped_ids_df.withColumn("group_row_num", floor(output_label_grouped_ids_df.group_row_num / lit(LABEL_BUCKET_JSON_SIZE)))

# Bucket all results into it's calculated buckets bucket
bucketed_label_df = output_label_grouped_ids_parts_df.orderBy("uniq_id").groupBy("label", "group_id", "group_row_num").agg(concat_ws("," ,collect_list(col("output"))).alias("output"))

# Create a unique id for each bucket, which we call the partiton id:
partion_ids_df = bucketed_label_df.withColumn("part_id", monotonically_increasing_id()).select(col("group_id").alias("f_group_id"), col("group_row_num").alias("f_group_row_num"), "part_id")

# join the part_id with the buckets
partioned_df = bucketed_label_df.join(partion_ids_df, (bucketed_label_df.group_id == partion_ids_df.f_group_id) & (bucketed_label_df.group_row_num == partion_ids_df.f_group_row_num)).drop("f_group_id", "f_group_row_num")

# COMMAND ----------

# Prepare a UDF that writes each bucket to storage
dbutils.fs.rm(os.path.join(OUTPUT_DIR, "json_output"), recurse=True)
dbutils.fs.mkdirs(os.path.join(OUTPUT_DIR, "json_output"))

def write_file(part_id, output):
  output_path = "/dbfs" + os.path.join(OUTPUT_DIR, "json_output", "{0}.json".format(part_id) )
  out_file = open(output_path, "w")
  out_file.write( "[{0}]".format(output))
  out_file.close()
  return part_id

write_output_udf = udf(lambda output: write_file(output[0], output[1]), StringType())
result_df = partioned_df.withColumn("part_id", write_output_udf(struct('part_id', 'output')))

# COMMAND ----------

# Force spark to invoke the previously defined UDF -> thus do the bad stuff -> materialize the whole dataset and execute udf
result_df.select("label", "part_id").write.mode("overwrite").parquet(os.path.join(OUTPUT_DIR, "label_output_mapping.parquet"))

# COMMAND ----------

# Read the previously created outputfile for post-processing (create inverted list)
label_output_mapping_df = spark.read.parquet(os.path.join(OUTPUT_DIR, "label_output_mapping.parquet"))

# COMMAND ----------

# Create the inverted list dataset
inv_list_df = label_output_mapping_df.orderBy(["label", asc("part_id")]).groupBy(col("label")).agg(concat_ws("," , collect_list(col("part_id")) ).alias("part_id_arr")).orderBy(["label"])

# COMMAND ----------

# Define UDF that transforms dataframe structure into a udf
def json_invl_transform(row):
  str_val = "\"{0}\": [{1}]".format(row.label, row.part_id_arr) 
  return [str_val]

output_rdd = inv_list_df.rdd.map(json_invl_transform)
inverted_list_df = output_rdd.toDF(["value"])

# COMMAND ----------

# Dense down the whole dataset into a single value (on a single machine with coalese -> potential bottleneck)
combined_df = inverted_list_df.coalesce(1).agg(concat_ws("," , collect_list(col("value")) ).alias("value"))

inv_list_output_value_df = combined_df.collect()

output_path = "/dbfs" + os.path.join(OUTPUT_DIR, "json_output", "inverted_list.json")
out_file = open(output_path, "w")
out_file.write( "{{{0}}}".format(inv_list_output_value_df[0].value))
out_file.close()