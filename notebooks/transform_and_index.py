# Databricks notebook source
# MAGIC %md
# MAGIC # Transform And Index Job
# MAGIC 
# MAGIC This job transforms the postprocessing output into a read- and processable output for the static search engine. It is divided into three general processing steps.
# MAGIC 
# MAGIC 1. Read and prepare input data.
# MAGIC   - Extract all classified images with a certain confidence (but at least one label will be kept regardless of the confidence level)
# MAGIC   - Add all other images that were downloaded but not classified, as not all images could be classified in the previous pipelines.
# MAGIC   - Calculate all "other labels" for each images, so we can output related labels of each image.
# MAGIC   - Calculate the buckets for each label (a bucket is a container partitioning the results for the search engine)
# MAGIC   - Result: intermediate_tandi parquet file
# MAGIC 2. Write Buckets To Disk
# MAGIC   - Calculate paritition ids
# MAGIC   - Clear output folder
# MAGIC   - Write buckets to output folder
# MAGIC   - Result: label_output_mapping parquet file
# MAGIC 3. Create Additional Output Data
# MAGIC   - Create Invertest List
# MAGIC   - Create Label List for autocompletion
# MAGIC   - Create a Label Cloud for the visualization
# MAGIC 4. Collect and Clean
# MAGIC   - Create tar archive from the output
# MAGIC   - Clean the json files on disk (not needed anymore)

# COMMAND ----------

import json
import os
import urllib.parse
from pyspark.sql.functions import desc, asc, monotonically_increasing_id, collect_list, col, dense_rank, row_number, lit,  floor, concat_ws, udf, struct, max, sort_array, round
from pyspark.sql.types import StructField, StringType, IntegerType, LongType, StructType, DateType, FloatType, DoubleType
from pyspark.sql.window import Window
import shutil

OUTPUT_DIR = "/mnt/group07/final_data_product/visualization/"
LABEL_BUCKET_JSON_SIZE = 10000
MINIMUM_CONFIDENCE = 5
PACKAGE_OUTPUT = True # Only for development?

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1. Read Input Data

# COMMAND ----------

# DBTITLE 1,Define helper functions
def get_at_least_one_label_per_img(input_df, min_confidence):
  img_id_window = Window.partitionBy(col('image_id'))
  # get for each image the label with the hightest confidence
  top_values_per_label_df = input_df.select("image_id", "confidence").groupBy("image_id").agg(max("confidence").alias("confidence"))
  top_labels_df = input_df.join(top_values_per_label_df, ["image_id", "confidence"]) # will result into multiple, if when ties exist
  top_labels_df = input_df.withColumn('max_confidence', max('confidence').over(img_id_window)).where(col('confidence') == col('max_confidence')).drop("max_confidence")
  # only use labels with confidence larger than MINIMUM_CONFIDENCE
  labels_with_min_confidence_df = input_df.where(col("confidence") > min_confidence)
  # create a union (from the min confidence and the top -> distinct to reduce if top val exist twice due to union)
  return labels_with_min_confidence_df.union(top_labels_df).distinct() 

def get_unclassified_images(input_flickr_df, input_labelled_images_ids_df):
  # Create "UNCLASSIFIED LABEL" for all images that not have been classified
  return input_flickr_df.select("image_id") \
                        .join(input_labelled_images_ids_df, 'image_id', 'left_anti') \
                        .withColumn('label',lit('unclassified')) \
                        .withColumn('confidence',lit(100.0))

def get_labels_with_numeric_ids(input_df):
  return input_df.select("label") \
          .orderBy("label") \
          .distinct() \
          .withColumn("label_num", row_number().over(Window.orderBy("label"))) \
          .withColumn("label_num", col("label_num") - lit(1)) 

# COMMAND ----------

# DBTITLE 1,Ingest raw data
# Read both the classified images + the flickr dataset
labels_df = spark.read.parquet("/mnt/group07/final_data_product/classification_result/labels.parquet")
# image_id|               label|          confidence|
downloaded_df = spark.read.parquet("/mnt/group07/final_data_product/classification_result/downloaded.parquet").limit(500000)
# image_id
flickr_metadata_df = spark.read.parquet("/mnt/group07/final_data_product/visualization_input.parquet").withColumnRenamed("id", "image_id")
# image_id|               title|   user_nsid|photo_video_farm_id|photo_video_server_id|photo_video_secret|photo_video_extension_original

# COMMAND ----------

# DBTITLE 1,Processing for Step 1.
label_with_min_top_df = get_at_least_one_label_per_img(labels_df, MINIMUM_CONFIDENCE)
# get labelled images ids:
labelled_images_ids_df = label_with_min_top_df.select("image_id").distinct()
# create a data product for all image ids and their labels
unclassified_label_df = get_unclassified_images(downloaded_df, labelled_images_ids_df)

all_imgs_df = label_with_min_top_df.union(unclassified_label_df)

# Reduce the images, acorrding to which have been downloaded:
all_imgs_df = all_imgs_df.join(downloaded_df, "image_id")

# Add a unique id, to persist the sort order, and then reorder for joins
labelled_imgs_wuid_df = all_imgs_df.orderBy([asc("label"), desc("confidence"), asc("image_id")]) \
                                   .withColumn("uniq_id", monotonically_increasing_id())

# Get a numeric id, that can be later accessed in the visualization for the related tags / tagged also as
labels_with_num_id = get_labels_with_numeric_ids(labelled_imgs_wuid_df)
labelled_imgs_wuid_with_label_id_df = labelled_imgs_wuid_df.join(labels_with_num_id, "label")

# Get all labels for an image, get product image_id, labels
# Translate all labels into the numeric id, and add to each image which other labels they belong to.
images_with_related_labels_df = labelled_imgs_wuid_with_label_id_df.select("image_id", "label_num").orderBy(["image_id", "label_num"]).groupBy("image_id").agg(collect_list(col("label_num")).alias("label_num_ids"))

labelled_imgs_wuid_with_other_labels_df = labelled_imgs_wuid_df.join(images_with_related_labels_df, "image_id")

# create a group number for each group
group_ids_df = labelled_imgs_wuid_with_other_labels_df.select("label").distinct().withColumn("group_id", row_number().over(Window.orderBy(col("label"))))

labelled_imgs_grouped_df = labelled_imgs_wuid_with_other_labels_df.join(group_ids_df, "label")

# number all items within a group with an index from 0 to n -> this might be slow, but we kind of need to preserver the order withing each group 🤔
labelled_imgs_grouped_rownum_df = labelled_imgs_grouped_df.withColumn("group_row_num", row_number().over(Window.partitionBy(col("label")).orderBy(col("uniq_id"))))
labelled_imgs_grouped_rownum_df = labelled_imgs_grouped_rownum_df.withColumn("group_row_num", col("group_row_num") - lit(1))

# Determine the amount of buckets by dividing the group internal numbering by the bucket size
labelled_imgs_grouped_rownum_parted_df = labelled_imgs_grouped_rownum_df.withColumn("group_row_num", floor(col("group_row_num") / lit(LABEL_BUCKET_JSON_SIZE)))

# COMMAND ----------

# DBTITLE 1,Materialize Step 01
# Persist this step:
labelled_imgs_grouped_rownum_parted_df.write.partitionBy("label").mode("overwrite").parquet(os.path.join(OUTPUT_DIR, "intermediate_tandi.parquet"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 02 - Write Buckets to Disk

# COMMAND ----------

# DBTITLE 1,Define Helper Functions
# import msgpack

"""
Transforms the row to the required JSON-representation.
"""
def json_transform(row):
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
  return [row.label, row.uniq_id, row.group_id, row.group_row_num, json.dumps(result_array, separators=(',', ':'))]

"""
Writes the partitioned JSON-Output to disk and returns the partition id as a result.
"""
def write_file(part_id, output):
  output_path = "/dbfs" + os.path.join(OUTPUT_DIR, "json_output", "{0}.json".format(part_id) )
  out_file = open(output_path, "w")
  out_file.write( "[{0}]".format(output))
  out_file.close()
  return part_id

def write_file_compressed(part_id, output):
  output_path = "/dbfs" + os.path.join(OUTPUT_DIR, "json_output_compressed", "{0}.json".format(part_id) )
  out_file = open(output_path, "w")
  output_string = "[{0}]".format(output)
  output_json = json.loads(output_string)
  output_json_compressed = msgpack.packb(output_json, use_bin_type=True)
  out_file.write(output_json_compressed)
  out_file.close()
  return part_id

# COMMAND ----------

# DBTITLE 1,Ingest Preprocessed Data from Step 1
prepared_labelled_imgs_df = spark.read.parquet(os.path.join(OUTPUT_DIR, "intermediate_tandi.parquet"))

# COMMAND ----------

# DBTITLE 1,Step 2 - Processing
# Add the flickr meta data to the result
prepped_imgs_with_meta_data_df = prepared_labelled_imgs_df.join(flickr_metadata_df, "image_id")

# Transform the metadata to a JSON output
labelled_imgs_output_df = prepped_imgs_with_meta_data_df.rdd.map(json_transform).toDF(['label', "uniq_id", "group_id", "group_row_num", "output"])

# Calculate a multiplier, to be able to calc a uniq id (TODO: possibly replaceable by mo. in. incr)
group_row_mult = len(str(prepared_labelled_imgs_df.select("group_row_num").groupBy().max().collect()[0][0]))

# Bucket all results into it's calculated buckets bucket
bucketed_df = labelled_imgs_output_df \
                .orderBy("uniq_id") \
                .groupBy("label", "group_id", "group_row_num") \
                .agg(concat_ws("," ,collect_list(col("output"))).alias("output")) \
                .withColumn("tmp_part_id",  (col("group_id") * int(10 ** group_row_mult)) + col("group_row_num")) \
                .select("label", "output", "tmp_part_id")

# Calculate final partition ids (will go to one single machine)
partition_ids_df = bucketed_df.select("tmp_part_id").withColumn("partition", row_number().over(Window.orderBy("tmp_part_id")))

# join the calculated partition with the buckets dataframe
partioned_df = bucketed_df.join(partition_ids_df, ["tmp_part_id"]).drop("tmp_part_id")

# Count the amount of images per label
imgs_per_label_count_df = prepared_labelled_imgs_df.groupBy("label").count().select("label", col("count").alias("img_cnt"))

# Prepare a UDF that writes each bucket to storage
write_output_udf = udf(lambda output: write_file(output.partition, output.output), LongType())
result_df = partioned_df.withColumn("partition", write_output_udf(struct('partition', 'output')))

# COMMAND ----------

# DBTITLE 1,Materialize Step 02
# Clean previously used folders, just in case there is some data left
dbutils.fs.rm(os.path.join(OUTPUT_DIR, "json_output"), recurse=True)
dbutils.fs.mkdirs(os.path.join(OUTPUT_DIR, "json_output"))

dbutils.fs.rm(os.path.join(OUTPUT_DIR, "json_output_compressed"), recurse=True)
dbutils.fs.mkdirs(os.path.join(OUTPUT_DIR, "json_output_compressed"))

# Materialize udf (and cause writing each file to disk!)
result_df.select("label", "partition").write.partitionBy("label").mode("overwrite").parquet(os.path.join(OUTPUT_DIR, "label_output_mapping.parquet"))

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 03 Create Additional Output Data

# COMMAND ----------

# DBTITLE 1,Define Helper Functions
# Define UDF that transforms dataframe structure into a udf
def inverted_list_row_to_json(row):
  str_val = "\"{0}\":{{ \"count\":{2}, \"buckets\":[{1}]}}".format(row.label, row.part_id_arr, row.img_cnt) 
  return [row.label, str_val]

# Write a array of all labels for autocompletition
def label_array_row_to_json(row):
    return ["\"{0}\"".format(row.label)]
  
# Define UDF that transforms dataframe structure into a udf
def word_cloud_row_to_json(row):
  str_val = "[\"{0}\",{1},{2}]".format(row.label, row.label_cloud_weight, row.img_cnt) 
  return [str_val]

def write_to_disk(output_path, output):
  out_file = open(output_path, "w")
  out_file.write(output)
  out_file.close()

# COMMAND ----------

# DBTITLE 1,Ingest Data from Step 2
# Read the previously created outputfile for post-processing (create inverted list)
label_output_mapping_df = spark.read.parquet(os.path.join(OUTPUT_DIR, "label_output_mapping.parquet")).orderBy(["label", "partition"])

# COMMAND ----------

# DBTITLE 1,Create and Write Inverted List
# Create the inverted list dataset
inv_list_df = label_output_mapping_df.orderBy(["label", asc("partition")]).groupBy(col("label")).agg(concat_ws("," , collect_list(col("partition")) ).alias("part_id_arr")).orderBy(["label"])

# Add the total count of images per label
inv_list_with_counts_df = inv_list_df.join(imgs_per_label_count_df, "label")

# Transform the dataset to a required JSON-representation
output_rdd = inv_list_with_counts_df.rdd.map(inverted_list_row_to_json)
inverted_list_df = output_rdd.toDF(["label", "value"])

# Collect the whole inverted list, and write to disk
inverted_list_output = inverted_list_df.coalesce(1).orderBy("label").select("value").agg(concat_ws("," , collect_list(col("value")) ).alias("value")).collect()[0].value

# Write to disk
write_to_disk(
  "/dbfs" + os.path.join(OUTPUT_DIR, "json_output", "inverted_list.json"),
  "{{{0}}}".format(inverted_list_output)
)

# COMMAND ----------

# DBTITLE 1,Create the Label List
# Get all tags
array_df = label_output_mapping_df.select("label").orderBy("label").distinct()

# Create the json output for each image and store as df
final_label_array_rdd = array_df.rdd.map(label_array_row_to_json)
final_label_array_df = final_label_array_rdd.toDF(["value"]).coalesce(1).agg(concat_ws("," , sort_array(collect_list(col("value"))) ).alias("value"))

# Write all available tags
write_to_disk(
  "/dbfs" + os.path.join(OUTPUT_DIR, "json_output", "tag_list.json"),
   "[{0}]".format(final_label_array_df.collect()[0].value)
)

# COMMAND ----------

# DBTITLE 1,Calculate Label Cloud
MAX_STEPS = 9
max_value = imgs_per_label_count_df.agg({"img_cnt": "max"}).collect()[0][0]
min_value = imgs_per_label_count_df.agg({"img_cnt": "min"}).collect()[0][0]
diff = max_value - min_value
step_divisor = diff / MAX_STEPS

label_cloud_df = imgs_per_label_count_df.withColumn("label_cloud_weight", round(inv_list_with_counts_df.img_cnt / step_divisor, 2) + lit(1)).select("label", "label_cloud_weight", "img_cnt")

label_cloud_json_df = label_cloud_df.rdd.map(word_cloud_row_to_json).toDF(["value"])

write_to_disk(
  "/dbfs" + os.path.join(OUTPUT_DIR, "json_output", "label_cloud.json"),
   "[{0}]".format(label_cloud_json_df.coalesce(1).agg(concat_ws("," , collect_list(col("value")) ).alias("value")).collect()[0].value)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## STEP 04 - Collect and Clean

# COMMAND ----------

if PACKAGE_OUTPUT:
  shutil.make_archive("/dbfs" + os.path.join(OUTPUT_DIR, "json_output"), 'tar', "/dbfs" + os.path.join(OUTPUT_DIR, "json_output"))

# COMMAND ----------

# Drop Raw Files
dbutils.fs.rm(os.path.join(OUTPUT_DIR, "json_output"), recurse=True)
dbutils.fs.rm(os.path.join(OUTPUT_DIR, "intermediate_tandi.parquet"), recurse=True)