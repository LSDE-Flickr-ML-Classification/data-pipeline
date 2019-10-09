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

from pyspark.sql.functions import monotonically_increasing_id
tags_wmeta_df = tags_wmeta_df.withColumn("id", monotonically_increasing_id())

# COMMAND ----------

from pyspark.sql.types import StructField, StringType, IntegerType, LongType, StructType, DateType, FloatType
from pyspark.sql.functions import collect_list
import json
import os
from pyspark.sql.functions import col

# tags_wmeta_df.write.mode("overwrite").partitionBy('label').json(os.path.join(OUTPUT_DIR, "tmp", "tags"))
# continue with scala - i suppose
def json_transform(row):
  result_dict = {
    "confidence": row.confidence,
    "image_id": "{0}".format(row.image_id),
    "flickr_url": "{0}".format(row.photo_video_page_url),
    "title": "{0}".format(row.title)
  }

  return [row.label, row.id, json.dumps(result_dict, sort_keys=True)]

output_rdd = tags_wmeta_df.rdd.map(json_transform)
output_label_df = output_rdd.toDF(['label', "id", "output"])

# COMMAND ----------

from pyspark.sql.functions import dense_rank, row_number, lit
from pyspark.sql.window import Window

# create a group number for each group
output_label_grouped_df = output_label_df.withColumn("group_id", dense_rank().over(Window.orderBy(output_label_df.label)))

# create unique group ids
output_label_grouped_ids_df = output_label_grouped_df.withColumn("group_row_num", row_number().over(Window.partitionBy(output_label_grouped_df.label).orderBy(output_label_grouped_df.id)))
output_label_grouped_ids_df = output_label_grouped_ids_df.withColumn("group_row_num", output_label_grouped_ids_df.group_row_num - lit(1))

# COMMAND ----------

from pyspark.sql.functions import floor, concat_ws
# Determine the amount of buckets
output_label_grouped_ids_parts_df = output_label_grouped_ids_df.withColumn("group_row_num", floor(output_label_grouped_ids_df.group_row_num / lit(100)))
# Bucket all results into it's bucket
bucketed_label_df = output_label_grouped_ids_parts_df.orderBy("id").groupBy("label", "group_id", "group_row_num").agg(concat_ws("," ,collect_list(col("output"))).alias("output"))
# Create a unique id for each bucket:
partion_ids_df = bucketed_label_df.withColumn("part_id", monotonically_increasing_id()).select(col("group_id").alias("f_group_id"), col("group_row_num").alias("f_group_row_num"), "part_id")

# join the part_id with the buckets
partioned_df = bucketed_label_df.join(partion_ids_df, (bucketed_label_df.group_id == partion_ids_df.f_group_id) & (bucketed_label_df.group_row_num == partion_ids_df.f_group_row_num)).drop("f_group_id", "f_group_row_num")

# COMMAND ----------

partioned_df.show(truncate=True)

# COMMAND ----------

from pyspark.sql.functions import udf, struct
from pyspark.sql.types import StringType
import os
import pathlib
  
dbutils.fs.rm(os.path.join(OUTPUT_DIR, "json_output"), recurse=True)

# create sub dirs (not sure if works that way)
#path = pathlib.Path(os.path.join(OUTPUT_DIR))
#path.parent.mkdir(parents=True, exist_ok=True)

dbutils.fs.mkdirs(os.path.join(OUTPUT_DIR, "json_output"))


def write_file(part_id, output):
  output_path = "/dbfs" + os.path.join(OUTPUT_DIR, "json_output", "{0}.json".format(part_id) )
  out_file = open(output_path, "w")
  out_file.write( "[{0}]".format(output))
  out_file.close()
  # return filename for inverted list (can only be the part id 🤔)
  return "{0}.json".format(part_id)

write_output_udf = udf(lambda output: write_file(output[0], output[1]), StringType())
result_df = partioned_df.withColumn("file_out", write_output_udf(struct('part_id', 'output')))

result_df.show() # this invokes the udf

# COMMAND ----------

file = open( "/dbfs" + os.path.join(OUTPUT_DIR, "json_output", "1.json") , "r") 
print(file.read())

# COMMAND ----------


# get the amount of partions:
#partions_num = partioned_df.count()
# assign each partion

# create a new rdd:
# partioned_df.select("part_id").show(100000)

#OUTFILE_PATH = os.path.join(OUTPUT_DIR, "txt", "sample")

#output_df.show()
# partitionBy(partions_num,lambda k: int(k[4]))



                     
#valid_img_download_url_udf(df_raw_flickr["photo_video_download_url"]))

#output_rdd.toDF().write.format("json").save(os.path.join(OUTPUT_DIR, "txt", "sample"))
# output_rdd.partitionBy(partions_num).saveAsTextFile(os.path.join(OUTPUT_DIR, "txt", "sample"))
# newpairRDD = pairRDD.partitionBy(5,lambda k: int(k[0]))


# final_part_id = partioned_df.partitionBy("part_id")

# 

#
# newpairRDD = pairRDD.partitionBy(5,lambda k: int(k[0]))

# tdf.partitionBy(5).toDF().write.format("json").save(os.path.join(OUTPUT_DIR, "txt", "sample"))
# tdf2 = tdf.toDF(['label', "output"]).groupBy(col("label")).agg(collect_list(col("output")).alias("output"))
  
  #  .write
  #.format("json")
  #.save("json/file/location/to/save")
#print()
#parts = tdf2.count()
#tdf2.rdd.partitionBy(parts).saveAsTextFile(os.path.join(OUTPUT_DIR, "txt", "sample"))

# COMMAND ----------

#from pyspark.sql import functions as F
# check: create folder for each tag
# check: write (firt approach) one big json for everything
# check: write chunked version
#       -> chunk big json for each tag into serveral small ones
# TODO: write inv list for all tags that were written (determine name of the chunks? ; )
# --> create df which maps tag to chunckname -> transform that to a json-baed inv list
 # TODO: to satify the app interface: write index.json

# COMMAND ----------

