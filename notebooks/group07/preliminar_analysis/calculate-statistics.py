# Databricks notebook source
# MAGIC %run /group07/shared

# COMMAND ----------

###
# Get the total count of the dataset
###
def get_total_count():
  files_list = dbutils.fs.ls("/mnt/data/flickr")
  for i, file in enumerate(files_list):
    df_bucket = spark.read.format("CSV").option("delimiter", "\t").schema(get_csv_data_scheme()).load(file.path[5:])
    total_count += df_bucket.count()
    print("{0} | {1}".format(file.name, df_bucket.count()))
  print("{0} | {1}".format("Total", total_count))
# uncomment and use
# get_total_count()

# COMMAND ----------

from functools import reduce

###
# Get the file sizes count of the dataset
###
files = dbutils.fs.ls("/mnt/data/flickr")
total_bytes = reduce(lambda acc, f: acc + f.size, files, 0)
print("total bytes: {0}".format(total_bytes))

# COMMAND ----------

# Get total row count of sample
df_raw_flickr = spark.read.parquet("/mnt/group07/full/flickr.parquet")
print("total rows: {0}".format(df_raw_flickr.count()))

# Get total byte size
df_raw_flickr.write.format("CSV").mode("overwrite").option("delimiter", "\t").save("/mnt/group07/analysis/sample.csv")
files = dbutils.fs.ls("/mnt/group07/analysis/sample.csv")
total_bytes = reduce(lambda acc, f: acc + f.size, files, 0)
print("total bytes: {0}".format(total_bytes))

# COMMAND ----------

print(len(df_raw_flickr.columns))
print(df_raw_flickr.columns)

# COMMAND ----------

# Get Byte Size of Relevant Columns:
df_raw_flickr_subset = df_raw_flickr.select("id", "user_tags", "machine_tags", "photo_video_page_url", "photo_video_download_url", "photo_video_marker", "title")
df_raw_flickr_subset.write.format("CSV").mode("overwrite").option("delimiter", "\t").save("/mnt/group07/analysis/sample_subset.csv")
files = dbutils.fs.ls("/mnt/group07/analysis/sample_subset.csv")
total_bytes_subset = reduce(lambda acc, f: acc + f.size, files, 0)
print("total subset bytes: {0}".format(total_bytes_subset))
print("ratio subset bytes: {0}".format(total_bytes_subset / total_bytes))

# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df_download_history = spark.read.parquet("/mnt/group07/download_history.parquet")
df_download_history.show(100, False)

# COMMAND ----------

df_status_code_mean = df_download_history.select("status_code", "download_duration").orderBy("status_code").groupBy("status_code").mean("download_duration")
df_status_code_mean.show()

# COMMAND ----------

df_status_code_count = df_download_history.orderBy("status_code").groupBy("status_code").count()
df_status_code_count.show()
df_scc_pandas = df_status_code_count.toPandas()

x = range(3)
x_labels = df_scc_pandas.status_code

# y_pos = np.array(str(df_scc_pandas.status_code))
#plt.bar(y_pos, height, color=(0.2, 0.4, 0.6, 0.6))

fig = plt.figure(figsize=(6,4))
bars = plt.bar(x, list(df_scc_pandas["count"]), color="#3A81BA")
plt.title('HTTP Return Codes Freq.')
plt.xticks(x, x_labels, ha='left')

# access the bar attributes to place the text in the appropriate location
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x(), yval + 100, yval)

# COMMAND ----------

display(fig)

# COMMAND ----------

from pyspark.sql.functions import col

df_sc_200_pd = df_download_history.select("status_code", col("download_duration").alias("Duration in seconds")).toPandas()

fig, ax = plt.subplots(figsize=(7,3))
bxp = df_sc_200_pd.boxplot(column=['Duration in seconds'], by='status_code', ax=ax)
print(df_sc_200_pd.groupby("status_code").describe())
plt.suptitle('')
plt.subplots_adjust(left=0.2, right=0.7, top=0.9, bottom=0.2)
bxp.set_xlabel("HTTP Status Code")
 

# COMMAND ----------

display(fig)

# COMMAND ----------

df_video_img_ratio = df_raw_flickr_subset.select("id", "photo_video_marker").orderBy("photo_video_marker").groupBy("photo_video_marker").count()
df_video_img_ratio.show()
df_video_img_ratio_pd = df_video_img_ratio.toPandas()

# COMMAND ----------

df_durations = df_download_history.select(df_download_history.download_duration)
df_durations.show()
df_durations_pd = df_durations.toPandas()
d = df_durations_pd['download_duration']
fig = plt.figure(figsize=(8,6))

n, bins, patches = plt.hist(x=d, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
# plt.title('My Very Own Histogram')
plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

# COMMAND ----------

display(fig)

# COMMAND ----------

fig = plt.figure(figsize=(8,6))


df_durations_pd.boxplot()

display(fig)

# COMMAND ----------

df_file_size_mean = df_download_history.select("image_size")
df_file_size_mean_pd = df_file_size_mean.toPandas()
df_file_size_mean.describe().show()
fig = plt.figure()
df_file_size_mean_pd.boxplot()


# COMMAND ----------

display(fig)

# COMMAND ----------

df_download_history.filter(df_download_history.image_size == 1907786).show(10, False)

# COMMAND ----------

df_download_history.filter(df_download_history.image_size == 1020).show(10, False)

# COMMAND ----------

df_pixel_sizes = df_download_history.select("image_px_width", "image_px_height")

df_ps_pd = df_pixel_sizes.toPandas()

fig = plt.figure()
plt.scatter(df_ps_pd["image_px_width"], df_ps_pd["image_px_height"])
plt.title("Scatter Plot for Sample")
plt.xlabel("Width in px")
plt.ylabel("Height in px")


# COMMAND ----------

display(fig)

# COMMAND ----------

# Determine the location based ratio:
df_raw_flickr = spark.read.parquet("/mnt/group07/full/flickr.parquet")

df_raw_flickr = df_raw_flickr.select("longitude", "latitude", "accuracy")

df_loc_parsed = df_raw_flickr \
              .withColumn("longitude", df_raw_flickr["longitude"].isNull()) \
              .withColumn("latitude", df_raw_flickr["latitude"].isNull()) \
              .withColumn("accuracy", df_raw_flickr["accuracy"].isNull())

df_loc_parsed.groupBy("longitude").count().show()
df_loc_parsed.groupBy("latitude").count().show()
df_loc_parsed.groupBy("latitude").count().show()