# Databricks notebook source
# DBTITLE 1,Read Classification Output
df = spark.read.option("header","true").csv("/mnt/group07/final_data_product/classified_images/0/part*.csv")

# COMMAND ----------

BUCKET_NUM = 7
files = dbutils.fs.ls("/mnt/group07/final_data_product/downloaded_images-{0}.parquet".format(BUCKET_NUM))
bytes_size = 0
for file in files:
  bytes_size += file.size
print("/mnt/group07/final_data_product/downloaded_images-{0}.parquet".format(BUCKET_NUM))
print(bytes_size)

# COMMAND ----------

df.select("id").distinct().count()

# COMMAND ----------

df.show()

# COMMAND ----------

#df_temp = spark.read.parquet("/mnt/group07/final_data_product/downloaded_images-1.parquet")
df_temp = spark.read.parquet("/mnt/group07/final_data_product/classified_images/9-cpu/images_classified.parquet")

# COMMAND ----------

df_temp.show(truncate=False)

# COMMAND ----------

res = df_temp.groupBy("error_code").count()

# COMMAND ----------

res.show(truncate=False)

# COMMAND ----------

list_img = df_temp.limit(100).collect()

# COMMAND ----------

from PIL import Image
import io
for img in list_img:
  if img.img_binary:
    pil = Image.open(io.BytesIO(img.img_binary))
    print(pil)

# COMMAND ----------

pil = Image.open(io.BytesIO(list_img[0].img_binary))

# COMMAND ----------

import matplotlib.pyplot as plt
plt.imshow(pil)
display(plt.show())

# COMMAND ----------

print(list_img[0].id)

# COMMAND ----------

dbutils.fs.ls("/mnt/group07/")

# COMMAND ----------



# COMMAND ----------

df_sample = spark.read.format("CSV").option("delimiter", "\t").load("/mnt/data/flickr/yfcc100m_dataset-0.bz2")

df_sample.printSchema()

# COMMAND ----------

df_res = df_sample.where(df_sample._c0 == "10000000")

# COMMAND ----------

df_res.show(truncate=False)