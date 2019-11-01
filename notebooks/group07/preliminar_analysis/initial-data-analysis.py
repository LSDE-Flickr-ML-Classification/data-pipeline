# Databricks notebook source
# MAGIC %run /group07/shared

# COMMAND ----------

###
# Sampels the flicker dataset with default seed lsde in ascii numbers
###
def sample_flickr_dataset(full_sample = False, prefix = "", seed = 108115100101):
  files_list = dbutils.fs.ls("/mnt/data/flickr")
  DF_SAMPLE_SIZE_FRAC = 0.0001
  BUCKET_COUNT = (len(files_list) if full_sample else 1) 

  df_sampled_buckets = spark.createDataFrame([], get_csv_data_scheme())
  print("file | sample size")
  for i, file in enumerate(files_list[0:BUCKET_COUNT]):
    df_bucket = spark.read.format("CSV").option("delimiter", "\t").schema(get_csv_data_scheme()).load(file.path[5:]).sample(True, DF_SAMPLE_SIZE_FRAC, seed=seed)
    df_bucket.write.format("parquet").save("/mnt/group07/{0}flickr_bucket_{1}.parquet".format(prefix, i))
    df_sampled_buckets = df_sampled_buckets.union(df_bucket)
    
    print("{0} | {1}".format(file.name, df_bucket.count()))
  df_sampled_buckets.write.format("parquet").save("/mnt/group07/{0}flickr.parquet".format(prefix))
  return df_sampled_buckets

# COMMAND ----------

def col_basic_analysis(col_name, df):
  df_col = df.select(col_name)
  col_total_cnt = df_col.count()
  null_values_cnt = df_col.filter(df_col[col_name].isNull()).count()  
  return [col_name, col_total_cnt, null_values_cnt]

# COMMAND ----------

from urllib.parse import urlparse

def is_flickr_image_download_url(url):
  try:
    result = urlparse(url)
    if (result.scheme != "http") :
      return False
    if (not result.path.endswith(".jpg")):
      return False
    return all([result.scheme, result.netloc, result.path])
  except ValueError:
    return False

# COMMAND ----------

import requests
from timeit import default_timer as timer
from datetime import timedelta
from PIL import Image
import io

def download_and_save_image(row):
  try:
    # return (row.id, None, None, None, None)
    # initiate download:
    start_download = timer()
    req = requests.get(row.photo_video_download_url)
    end_download = timer()
    download_in_seconds = timedelta(seconds=end_download-start_download).total_seconds()

    status_code = req.status_code
    # process the result
    if req.status_code == 200:
      image_in_bytes = len(req.content)
      image_file_path = "/dbfs/mnt/group07/images_stats/{0}.jpg".format(row.id)
      
      image = Image.open(io.BytesIO(req.content))
      image_height = image.size[1]
      image_width = image.size[0]
       
      f = open(image_file_path,'wb')
      f.write(req.content)
      f.close()
    else:
        image_in_bytes = None
        image_file_path = None
        image_height = None
        image_width = None
  except Exception as e:
    return (row.id, None, None, None, None, None, str(e))
  return (row.id, status_code, download_in_seconds, image_in_bytes, image_width, image_height, image_file_path)

# COMMAND ----------

# Triggers the flickr download based on the delta of images
def batch_download_images(df_download_links, df_download_hist):
  batch_map = df_download_links.join(df_download_hist, "id", "leftanti").rdd.map(download_and_save_image)
  return spark.createDataFrame(batch_map, get_download_history_schema())

# Returns a download history dataframe (either from disk or new one)
def load_download_history():
  # open a new download history
  if "download_history.parquet/" in [fi.name for fi in dbutils.fs.ls("/mnt/group07")]:
    return spark.read.parquet("/mnt/group07/download_history.parquet")
  else:
    return spark.createDataFrame([], get_download_history_schema())

# COMMAND ----------

df_raw_flickr = spark.read.parquet("/mnt/group07/full/flickr.parquet")
print(df_raw_flickr.count())

# COMMAND ----------

from pyspark.sql.functions import udf

valid_img_download_url_udf = udf(lambda str_url: str_url if is_flickr_image_download_url(str_url) else None, StringType())

df_flickr = df_raw_flickr \
            .withColumn("id", df_raw_flickr["id"].cast(LongType())) \
            .withColumn("date_taken", df_raw_flickr["date_taken"].cast(DateType())) \
            .withColumn("date_uploaded", df_raw_flickr["date_uploaded"].cast(IntegerType())) \
            .withColumn("longitude", df_raw_flickr["longitude"].cast(IntegerType())) \
            .withColumn("latitude", df_raw_flickr["latitude"].cast(IntegerType())) \
            .withColumn("accuracy", df_raw_flickr["accuracy"].cast(IntegerType())) \
            .withColumn("photo_video_download_url", valid_img_download_url_udf(df_raw_flickr["photo_video_download_url"])) \
            .withColumn("photo_video_server_id", df_raw_flickr["photo_video_server_id"].cast(IntegerType())) \
            .withColumn("photo_video_farm_id", df_raw_flickr["photo_video_farm_id"].cast(IntegerType())) \
            .withColumn("photo_video_marker", df_raw_flickr["photo_video_marker"].cast(IntegerType()))

df_flickr.printSchema()

# COMMAND ----------

df_id_download_link = df_flickr.where(df_flickr.photo_video_marker == 0).select(df_flickr.id, df_flickr.photo_video_download_url)

# Create required dirs
dbutils.fs.mkdirs("/mnt/group07/images_stats/")

# Load the download history (to avoid downloading unecessary stuff)
df_download_history = load_download_history()

# Execute the Batch download images function, that pulls the delta history from flickr
df_downloads = batch_download_images(df_id_download_link, df_download_history)
df_download_history = df_download_history.union(df_downloads)

# write the latest download history
df_download_history.write.mode("overwrite").format("parquet").save("/mnt/group07/download_history.parquet")

# COMMAND ----------

df_download_history = spark.read.parquet("/mnt/group07/download_history.parquet")
df_download_history.show(100, False)

# COMMAND ----------

df = df_download_history.select("status_code").groupBy("status_code").count()
df.show()

# COMMAND ----------

df_download_history.filter(df_download_history.status_code == 200).show(10000, False)

# COMMAND ----------

df_download_history.printSchema()

# COMMAND ----------

df_download_history.select("image_size").groupBy().sum().show()

# COMMAND ----------

spark.read.parquet("/mnt/group07/download_history.parquet").show(truncate=False)

# COMMAND ----------

# DBTITLE 1,Show Overview on Image Resolutions

import pyspark.sql.functions as pf
import matplotlib.pyplot as plt
import pandas as pd

# Collect the relevant data
df_width_height = spark.read.parquet("/mnt/group07/download_history.parquet").select(pf.col("image_px_width").alias("img_width"), pf.col("image_px_height").alias("img_height"))
df_resolutions = df_width_height.where(df_width_height.img_width.isNotNull()).where(df_width_height.img_height.isNotNull())
pd_img_width_df = df_resolutions.select("img_width", "img_height").toPandas()

# Plot
plt.close("all")
fig3 = plt.figure(constrained_layout=True, figsize=(4, 8))
gs = fig3.add_gridspec(8, 4)

f3_ax1 = fig3.add_subplot(gs[0:4, :])
f3_ax2 = fig3.add_subplot(gs[4:8, 0:2])
f3_ax3 = fig3.add_subplot(gs[4:8, 2:4])


f3_ax1.scatter(x="img_width", y="img_height", data=pd_img_width_df, s=1, edgecolors='b')
f3_ax1.grid(color='grey', linestyle='dashed', linewidth=0.5)
f3_ax1.set_title('Width And Height Distribution')
f3_ax1.set_xlabel('Height')
f3_ax1.set_ylabel('Width')

f3_ax2.boxplot(x=pd_img_width_df["img_width"], showfliers=False)
f3_ax2.grid(color='grey', linestyle='dashed', linewidth=0.5)
f3_ax2.set_title('Width')
f3_ax2.set_xlabel('Number of Pixel')
f3_ax2.set_ylabel('')
f3_ax2.set_ylim([497,502])
f3_ax2.set_xticklabels([])

f3_ax3.boxplot(x=pd_img_width_df["img_height"], showfliers=False)
f3_ax3.grid(color='grey', linestyle='dashed', linewidth=0.5)
f3_ax3.set_title('Height')
f3_ax3.set_xlabel('Number of Pixel')
f3_ax3.set_ylabel('')
f3_ax3.set_ylim([80,520])
f3_ax3.set_xticklabels([])
f3_ax3.set_title('Height')


fig3.savefig("/dbfs/mnt/group07/plots/initial_resolutions.pdf")
display(fig3)

# COMMAND ----------

# DBTITLE 1,Plot Boxplots for Request Times
import statistics 

plt.close("all")
pd_df = spark.read.parquet("/mnt/group07/download_history.parquet").select("status_code", pf.col("download_duration").alias("Duration in seconds")).toPandas()
fig, ax = plt.subplots(figsize=(5,5))

pdfg = pd_df.groupby('status_code')

mean_dict = pd_df.groupby('status_code').mean().to_dict()
med_dict = pd_df.groupby('status_code').median().to_dict()
cax = pd_df.boxplot(by='status_code', ax=ax)
cax.set_xticklabels(['HTTP {0!s}\n$n$={1} \n mean={2}\n med={3}'.format(k, len(v), round(mean_dict["Duration in seconds"][k], 2), round(med_dict["Duration in seconds"][k], 2)) for k, v in pdfg])
plt.suptitle('')

ax.set_xlabel("HTTP Status Code")
fig.savefig("/dbfs/mnt/group07/plots/initial_downloads.pdf")
display(fig)

# COMMAND ----------

# DBTITLE 1,Calculate Statistics on File Size
df_file_size_mean = spark.read.parquet("/mnt/group07/download_history.parquet").select("image_size")
df_file_size_mean_pd = df_file_size_mean.toPandas()
df_file_size_mean.describe().show()