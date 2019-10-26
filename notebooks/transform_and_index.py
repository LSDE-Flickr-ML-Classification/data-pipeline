# Databricks notebook source
import json
import os
import urllib.parse
from pyspark.sql.functions import desc, asc, monotonically_increasing_id, collect_list, col, dense_rank, row_number, lit,  floor, concat_ws, udf, struct
from pyspark.sql.types import StructField, StringType, IntegerType, LongType, StructType, DateType, FloatType, DoubleType
from pyspark.sql.window import Window

OUTPUT_DIR = "/mnt/group07/final_data_product/visualization/"
LABEL_BUCKET_JSON_SIZE = 100
MINIMUM_CONFIDENCE = 5
PACKAGE_OUTPUT = True # Only for development?

# COMMAND ----------

# DBTITLE 1,Read Input Data
# Read both the classified images + the flickr dataset
labels_df = spark.read.parquet("/mnt/group07/final_data_product/classification_result/labels.parquet")
# image_id|               label|          confidence|
downloaded_df = spark.read.parquet("/mnt/group07/final_data_product/classification_result/downloaded.parquet").limit(500000)
# image_id
flickr_metadata_df = spark.read.parquet("/mnt/group07/final_data_product/visualization_input.parquet").withColumnRenamed("id", "image_id")
# image_id|               title|   user_nsid|photo_video_farm_id|photo_video_server_id|photo_video_secret|photo_video_extension_original

# COMMAND ----------

# Filter Meta Data so that we only include the images that we were able to download
flickr_metadata_df = flickr_metadata_df.join(downloaded_df, 'image_id', 'inner')

# COMMAND ----------

import pyspark.sql.functions as f
from pyspark.sql import Window

w = Window.partitionBy(col('image_id'))
top_labels_df = labels_df.withColumn('max_confidence', f.max('confidence').over(w)).where(f.col('confidence') == f.col('max_confidence')).drop("max_confidence")

# COMMAND ----------

# only use labels with confidence larger than MINIMUM_CONFIDENCE
label_with_min_df = labels_df.where(col("confidence") > MINIMUM_CONFIDENCE)

# COMMAND ----------

label_with_min_top_df = label_with_min_df.union(top_labels_df).distinct() # create a union (from the min confidence and the top -> distinct to reduce if top val exist twice due to union)

# COMMAND ----------

# get labelled images ids:
labelled_images_ids_df = label_with_min_top_df.select("image_id").distinct()

# COMMAND ----------

# Create "UNCLASSIFIED LABEL" for all images that not have been classified
unclassified_label_df = flickr_metadata_df.select("image_id").join(labelled_images_ids_df, 'image_id', 'left_anti')
unclassified_label_df = unclassified_label_df.withColumn('label',lit('unclassified')).withColumn('confidence',lit(100.0))

# COMMAND ----------

# create a data product for all image ids and their labels
labelled_imgs_df = label_with_min_df.union(unclassified_label_df)

# COMMAND ----------

from pyspark.sql.functions import lower, col, round, regexp_replace

labelled_imgs_preprocessed_df = labelled_imgs_df \
                                  .withColumn("label",f.lower(f.col("label"))) \
                                  .withColumn('label', regexp_replace('label', '_', ' ')) \
                                  .withColumn("confidence", round(col("confidence"), 2))

# COMMAND ----------

labelled_imgs_preprocessed_df = labelled_imgs_preprocessed_df.join(flickr_metadata_df, "image_id")

# Order labelling results for inverted list conversion / search index
labelled_imgs_preprocessed_df = labelled_imgs_preprocessed_df.orderBy([asc("label"), desc("confidence"), asc("image_id")])


# COMMAND ----------

# Add a uniq id to remember the correct ordering
labelled_imgs_wuid_df = labelled_imgs_preprocessed_df.withColumn("uniq_id", monotonically_increasing_id())

# COMMAND ----------

labels_with_num_id = labelled_imgs_wuid_df.select("label").orderBy("label").distinct().withColumn("label_num", row_number().over(Window.orderBy("label"))).withColumn("label_num", col("label_num") - lit(1))

# COMMAND ----------

labelled_imgs_wuid_with_label_id_df = labelled_imgs_wuid_df.join(labels_with_num_id, "label")

# COMMAND ----------

# Get all labels for an image, get product image_id, labels
image_id_with_labels_df = labelled_imgs_wuid_with_label_id_df.select("image_id", "label_num").orderBy(["image_id", "label_num"]).groupBy("image_id").agg(collect_list(col("label_num")).alias("label_num_ids"))

# COMMAND ----------

labelled_imgs_wuid_with_other_labels_df = labelled_imgs_wuid_df.join(image_id_with_labels_df, "image_id")

# COMMAND ----------

# Create the JSON Object for each item in the bucket
import urllib.parse

def json_transform(row):
  # image_id|               title|    user_nsid|photo_video_farm_id|photo_video_server_id|photo_video_secret|photo_video_extension_original
  result_array = [
    int(row.image_id),
    "{0}".format(row.user_nsid),
    int(row.photo_video_farm_id),
    int(row.photo_video_server_id),
    "{0}".format(row.photo_video_secret),
    "{0}".format(row.photo_video_extension_original),
    row.confidence,
    row.label_num_ids,
  ]

  # use custom seperator to save some space
  return [row.label, row.uniq_id, json.dumps(result_array, separators=(',', ':'))]

# Create the json output for each image and store as df
labelled_imgs_output_df = labelled_imgs_wuid_with_other_labels_df.rdd.map(json_transform).toDF(['label', "uniq_id", "output"])

# COMMAND ----------

# create a group number for each group
labelled_imgs_output_grouped_df = labelled_imgs_output_df.withColumn("group_id", dense_rank().over(Window.orderBy(col("label"))))

# number all items within a group with an index from 0 to n
labelled_imgs_output_grouped_rownum_df = labelled_imgs_output_grouped_df.withColumn("group_row_num", row_number().over(Window.partitionBy(col("label")).orderBy(col("uniq_id"))))
labelled_imgs_output_grouped_rownum_df = labelled_imgs_output_grouped_rownum_df.withColumn("group_row_num", col("group_row_num") - lit(1))

# COMMAND ----------

# Determine the amount of buckets by dividing the group internal numbering by the bucket size
labelled_imgs_output_grouped_rownum_parted_df = labelled_imgs_output_grouped_rownum_df.withColumn("group_row_num", floor(col("group_row_num") / lit(LABEL_BUCKET_JSON_SIZE)))

# Bucket all results into it's calculated buckets bucket
bucketed_df = labelled_imgs_output_grouped_rownum_parted_df.orderBy("uniq_id").groupBy("label", "group_id", "group_row_num").agg(concat_ws("," ,collect_list(col("output"))).alias("output"))

# Create a unique id for each bucket, which we call the partiton id:
partition_ids_df = bucketed_df.withColumn("part_id", monotonically_increasing_id()).select(col("group_id").alias("f_group_id"), col("group_row_num").alias("f_group_row_num"), "part_id")

# join the part_id with the buckets
partioned_df = bucketed_df.join(partition_ids_df, (bucketed_df.group_id == partition_ids_df.f_group_id) & (bucketed_df.group_row_num == partition_ids_df.f_group_row_num)).drop("f_group_id", "f_group_row_num")

# COMMAND ----------

from pyspark.sql.functions import count, lit
# Count all instances
imgs_per_label_count_df = labelled_imgs_output_grouped_rownum_df.groupBy("label").count().select("label", col("count").alias("img_cnt"))

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

# Write the dataset to materialize the udf
result_df.select("label", "part_id").write.mode("overwrite").parquet(os.path.join(OUTPUT_DIR, "label_output_mapping.parquet"))

# COMMAND ----------

# Read the previously created outputfile for post-processing (create inverted list)
label_output_mapping_df = spark.read.parquet(os.path.join(OUTPUT_DIR, "label_output_mapping.parquet"))
label_output_mapping_df.count()

# COMMAND ----------

label_output_mapping_df.show()

# COMMAND ----------

# Write a array of all labels for autocompletition
def json_prepare(row):
    return ["\"{0}\"".format(row.label)]

# Get all tags
array_df = label_output_mapping_df.select("label").orderBy("label").distinct()

# Create the json output for each image and store as df
final_label_array_rdd = array_df.rdd.map(json_prepare)
final_label_array_df = final_label_array_rdd.toDF(["value"]).coalesce(1).agg(concat_ws("," , collect_list(col("value")) ).alias("value"))

# COMMAND ----------

# Write all available tags
output_path = "/dbfs" + os.path.join(OUTPUT_DIR, "json_output", "tag_list.json")
out_file = open(output_path, "w")
out_file.write( "[{0}]".format(final_label_array_df.collect()[0].value))
out_file.close()

# COMMAND ----------

# Create the inverted list dataset
inv_list_df = label_output_mapping_df.orderBy(["label", asc("part_id")]).groupBy(col("label")).agg(concat_ws("," , collect_list(col("part_id")) ).alias("part_id_arr")).orderBy(["label"])

# COMMAND ----------

inv_list_with_counts_df = inv_list_df.join(imgs_per_label_count_df, "label")

# COMMAND ----------

# Define UDF that transforms dataframe structure into a udf
def json_invl_transform(row):
  str_val = "\"{0}\":{{ \"count\":{2}, \"buckets\":[{1}]}}".format(row.label, row.part_id_arr, row.img_cnt) 
  return [row.label, str_val]

output_rdd = inv_list_with_counts_df.rdd.map(json_invl_transform)
inverted_list_df = output_rdd.toDF(["label", "value"])

# COMMAND ----------

# Dense down the whole dataset into a single value (on a single machine with coalese -> potential bottleneck)
combined_df = inverted_list_df.coalesce(1).orderBy("label").select("value").agg(concat_ws("," , collect_list(col("value")) ).alias("value"))

inv_list_output_value_df = combined_df.collect()

output_path = "/dbfs" + os.path.join(OUTPUT_DIR, "json_output", "inverted_list.json")
out_file = open(output_path, "w")
out_file.write( "{{{0}}}".format(inv_list_output_value_df[0].value))
out_file.close()

# COMMAND ----------

# DBTITLE 1,Calculate Label Cloud
MAX_STEPS = 9
max_value = imgs_per_label_count_df.agg({"img_cnt": "max"}).collect()[0][0]
min_value = imgs_per_label_count_df.agg({"img_cnt": "min"}).collect()[0][0]
diff = max_value - min_value
step_divisor = diff / MAX_STEPS

# COMMAND ----------

from pyspark.sql.functions import round

label_cloud_df = imgs_per_label_count_df.withColumn("label_cloud_weight", round(inv_list_with_counts_df.img_cnt / step_divisor, 2) + lit(1)).select("label", "label_cloud_weight", "img_cnt")

# COMMAND ----------

# Define UDF that transforms dataframe structure into a udf
def json_label_cloud_transform(row):
  str_val = "[\"{0}\",{1},{2}]".format(row.label, row.label_cloud_weight, row.img_cnt) 
  return [str_val]

label_cloud_json_df = label_cloud_df.rdd.map(json_label_cloud_transform).toDF(["value"])

# COMMAND ----------

# Dense down the whole dataset into a single value (on a single machine with coalese -> potential bottleneck)
write_df = label_cloud_json_df.coalesce(1).agg(concat_ws("," , collect_list(col("value")) ).alias("value"))

write_collected = write_df.collect()

output_path = "/dbfs" + os.path.join(OUTPUT_DIR, "json_output", "label_cloud.json")
out_file = open(output_path, "w")
out_file.write( "[{0}]".format(write_collected[0].value))
out_file.close()

# COMMAND ----------

# Package for easier download
import shutil
if PACKAGE_OUTPUT:
  shutil.make_archive("/dbfs" + os.path.join(OUTPUT_DIR, "json_output"), 'tar', "/dbfs" + os.path.join(OUTPUT_DIR, "json_output"))

# COMMAND ----------

