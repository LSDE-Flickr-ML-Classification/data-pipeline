# Databricks notebook source
import requests
import io
import os
from pyspark.sql.functions import col
from timeit import default_timer as timer
from datetime import timedelta
from torchvision import transforms
from PIL import Image

# COMMAND ----------

# OUTPUT_FILE = "/mnt/group07/final_data_product/downloaded_images-1.parquet"
OUTPUT_FILE = getArgument("OUTPUT_FILE", "")
# INPUT_FILE = "/mnt/group07/final_data_product/classification_input.parquet"
INPUT_FILE = getArgument("INPUT_FILE", "")

# COMMAND ----------

class ImageDownloader:
    transform_function = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224)
    ])

    @staticmethod
    def transform_image(image_bytes):
        image = Image.open(image_bytes)
        transformed_image = ImageDownloader.transform_function(image)
        return transformed_image
    

# COMMAND ----------

def download_and_save_image(row):
  image_bytes = None
  image_height = None
  image_width = None
  image_bytes_size = None
  
  if row.longitude is not None:
    try:
      img_s3_path = os.path.join("/dbfs/mnt/group01/images", "{0}_{1}_{2}.{3}".format(row.id, row.longitude, row.latitude, row.photo_video_extension_original) )
      s3_img_file = open(img_s3_path, "rb")
      img_read = s3_img_file.read()
      s3_img_file.close()
      # get image stats:
      image_bytes = bytearray(img_read)

      # resize image
      image_bytes_size = len(img_read)
      image = Image.open(io.BytesIO(image_bytes))
      image_height = image.size[1]
      image_width = image.size[0]
      
      img = ImageDownloader.transform_image(io.BytesIO(image_bytes))
      buffer = io.BytesIO()
      #buffer.seek(0)
      img.save(buffer, format='PNG')
      image_bytes = bytearray(buffer.getvalue())
      
      return (row.id, image_bytes, image_bytes_size, image_height, image_width, "s3")
    except Exception as e:
      # do nothing, and retry on real flickr data
      pass
  
  # try to download:
  try:
    req = requests.get(row.photo_video_download_url)
    if req.status_code == 200:
      image_bytes = bytearray(req.content)
      image_bytes_size = len(req.content)
      # resize image
      image = Image.open(io.BytesIO(image_bytes))
      image_height = image.size[1]
      image_width = image.size[0]
      
      img = ImageDownloader.transform_image(io.BytesIO(image_bytes))
      buffer = io.BytesIO()
      #buffer.seek(0)
      img.save(buffer, format='PNG')
      image_bytes = bytearray(buffer.getvalue())

    else:
      image_bytes = None
    return (row.id, image_bytes, image_bytes_size, image_height, image_width, req.status_code)
  except Exception as e:
    s = str(e)
    return (row.id, None, None, None, None, s)

# COMMAND ----------

# Start Timing
start_download = timer()

# COMMAND ----------

# DBTITLE 1,Whole Download Pipeline
from pyspark.sql.types import StructField, BinaryType, StructType, StringType, IntegerType

# Define Schema for output table
schema = StructType(fields=[
    StructField('id', StringType(), True),
    StructField('img_binary', BinaryType(), True),
    StructField('img_size', IntegerType(), True),  
    StructField('img_width', IntegerType(), True),
    StructField('img_height', IntegerType(), True),
    StructField('error_code', StringType(), True),
  ])


df_dl_links = spark.read.parquet(INPUT_FILE)

print("About to process {0} rows".format(df_dl_links.count()))
# Sort the dataframe along id
df_dl_links = df_dl_links.sort("id")
# Load the previously downloaded images (if job restarted)
if os.path.isdir("/dbfs" + OUTPUT_FILE):
  download_history_df = spark.read.parquet(OUTPUT_FILE)
else:
  download_history_df = spark.createDataFrame([], schema=schema)

# Determine the rows that have been downloaded already + determine the rows to be downloaded again
already_downloaded_ids = download_history_df.where(col("img_binary").isNotNull()).select("id")
to_download_df = df_dl_links.join(already_downloaded_ids, 'id', 'left_anti')
already_downloaded = already_downloaded_ids.count()
if (already_downloaded > 0):
  print("Restarted from {0} rows - downloading {1} rows".format(already_downloaded, to_download_df.count()))

# COMMAND ----------

batch_map = to_download_df.rdd.map(download_and_save_image)
final_df = spark.createDataFrame(batch_map, schema=schema)
# Force Evaluation of UDF
# download_count = final_df.rdd.count()
# print("Processed {0} rows".format(download_count))

# COMMAND ----------

# Create the new download file from previous downloads + new downloads
download_history_df = download_history_df.where(col("img_binary").isNotNull()).union(final_df)

# Write results to storage
download_history_df.write.format("parquet").save(OUTPUT_FILE)

# COMMAND ----------

# Finish timer
end_download = timer()
download_in_seconds = timedelta(seconds=end_download-start_download).total_seconds()
print(download_in_seconds)

# COMMAND ----------

# df_temp = spark.read.parquet("/mnt/group07/final_data_product/downloaded_images.parquet")

# COMMAND ----------

