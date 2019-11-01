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

# OUTPUT_FILE = ""
BUCKET_NUM = getArgument("NUM", "1")
OUTPUT_FILE = "/mnt/group07/final_data_product/classified_images/{0}-cpu/images_classified.parquet".format(BUCKET_NUM)
# INPUT_FILE = ""
INPUT_FILE = "/mnt/group07/final_data_product/download_input-{0}.parquet".format(BUCKET_NUM)
print("reading from {0}".format(INPUT_FILE))
print("writing to {0}".format(OUTPUT_FILE))

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

# Start Timing
start_download = timer()

# COMMAND ----------

from torch.utils.data import Dataset
from torchvision import models
from itertools import islice
from PIL import Image
from torchvision import transforms

import time
import torch
import os
import json
import io

MEAN = [0.485, 0.456, 0.406]
STANDARD_DEVIATION = [0.229, 0.224, 0.225]
IMAGE_TRANSFORM_FUNCTION = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STANDARD_DEVIATION)
])

# COMMAND ----------

print("Initializing the classification model")
model = models.inception_v3(pretrained=True)
model.eval()
model.to("cpu")

# COMMAND ----------

classes_list = json.load(open("/dbfs/mnt/group07/imagenet_classes.json"))
imagenet_class =[classes_list[str(k)][1] for k in range(len(classes_list))]
max_labels = 5

# COMMAND ----------

from torchvision import transforms

def classify_image(img_binary):
  if img_binary is None:
    return [[], "img data missing"]
  try:
    image_bytes = bytearray(img_binary)
    image = Image.open(io.BytesIO(image_bytes))
    image_tensor = IMAGE_TRANSFORM_FUNCTION(image)
    input_batch = image_tensor.unsqueeze(0)

    if torch.cuda.is_available():
      input_batch = input_batch.to('cuda')
      model.to('cuda')
    with torch.no_grad():
      output = model(input_batch)
    prediction = output[0]
    percentage = torch.nn.functional.softmax(prediction, dim=0) * 100
    _, indices = torch.sort(prediction, descending=True)
    matched_classes = [ (imagenet_class[idx], percentage[idx].item()) for idx in islice(indices, max_labels)]
  except Exception as e:
    return [[], str(e)]
    
  return [matched_classes, None]

# COMMAND ----------

def download_and_save_image(row):
  image_bytes = None
  image_height = None
  image_width = None
  image_bytes_size = None
  classes = []
  
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
        
      classes = result[0]
      status = result[1]
      if (status is  None):
        status = "s3"
      else:
        # append classification status
        status = "s3 - " + status
      
      return (row.id, classes, image_bytes_size, image_height, image_width, status)
    except Exception as e:
      # do nothing, and retry on real flickr data
      pass
  
  # try to download:
  try:
    req = requests.get(row.photo_video_download_url)
    status = req.status_code
    if req.status_code == 200:
      image_bytes = bytearray(req.content)
      image_bytes_size = len(req.content)
      # resize image
      image = Image.open(io.BytesIO(image_bytes))
      image_height = image.size[1]
      image_width = image.size[0]
      
      img = ImageDownloader.transform_image(io.BytesIO(image_bytes))
      buffer = io.BytesIO()
      
      img.save(buffer, format='PNG')
      image_bytes = bytearray(buffer.getvalue())
      
      result = classify_image(image_bytes)
      classes = result[0]
      if (result[1] is not None):
        # append classification status
        status += " {0}".format(result[1])

    else:
      image_bytes = None
    return (row.id, classes, image_bytes_size, image_height, image_width, status)
  except Exception as e:
    s = str(e)
    return (row.id, [], None, None, None, s)

# COMMAND ----------

# DBTITLE 1,Whole Download Pipeline
from pyspark.sql.types import StructField, BinaryType, StructType, StringType, IntegerType, ArrayType, DoubleType

# Define Schema for output table
schema = StructType(fields=[
    StructField('id', StringType(), True),
    #StructField('img_binary', BinaryType(), True),
     StructField("classes", ArrayType(
        StructType([
            StructField("label", StringType()),
            StructField("confidence", DoubleType())
        ])
     )),
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
already_downloaded_ids = download_history_df.where(col("img_width").isNotNull()).select("id")
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
download_history_df = download_history_df.where(col("img_width").isNotNull()).union(final_df)

#download_history_df.show(100)
# Write results to storage
download_history_df.write.format("parquet").save(OUTPUT_FILE)

# COMMAND ----------

# Finish timer
end_download = timer()
download_in_seconds = timedelta(seconds=end_download-start_download).total_seconds()
print(download_in_seconds)

# COMMAND ----------

df_temp = spark.read.parquet(OUTPUT_FILE)

# COMMAND ----------

df_temp.show(truncate=True)

# COMMAND ----------

