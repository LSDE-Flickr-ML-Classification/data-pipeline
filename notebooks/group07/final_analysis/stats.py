# Databricks notebook source


# COMMAND ----------

stats_df = spark.read.parquet("/mnt/group07/final_data_product/classification_result/stats.parquet")

# COMMAND ----------

downloaded_df = spark.read.parquet("/mnt/group07/final_data_product/classification_result/downloaded.parquet")

# COMMAND ----------

downloaded_df.distinct().count()

# COMMAND ----------

# Collect the relevant data
df_width_height = spark.read.parquet("/mnt/group07/final_data_product/classification_result/stats.parquet")
df_resolutions = df_width_height.where(df_width_height.img_width.isNotNull()).where(df_width_height.img_height.isNotNull())
df_resolutions.count()

# COMMAND ----------

downloaded_df.count()

# COMMAND ----------

downloaded_df.join().groupBy("error_code").count().show()

# COMMAND ----------

df = df_resolutions.join(downloaded_df.select(downloaded_df.image_id.alias("id")), "id", "leftanti")

# COMMAND ----------

df.show()

# COMMAND ----------

labels_df = spark.read.parquet("/mnt/group07/final_data_product/classification_result/labels.parquet")

# COMMAND ----------

labels_df.select("image_id").distinct().count()

# COMMAND ----------

# Get Total Amount Of Bytes Downloaded / S3 Access
stats_df.select("img_size").groupBy().sum().show()

# COMMAND ----------

import pyspark.sql.functions as pf
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Collect the relevant data
df_width_height = spark.read.parquet("/mnt/group07/final_data_product/classification_result/stats.parquet")
df_resolutions = df_width_height.where(df_width_height.img_width.isNotNull()).where(df_width_height.img_height.isNotNull())
df_resolutions.join(df, "id", "left_anti").select("img_size", "img_width", "img_height").describe().show()

# COMMAND ----------

import numpy as np                                                               
import matplotlib.pyplot as plt

def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                '%d' % int(height),
                ha='center', va='bottom')
        
        
plt.close("all")
count_list = spark.read.parquet("/mnt/group07/final_data_product/classification_result/stats.parquet").select("error_code").groupBy("error_code").count().collect()
labels, values = zip(*count_list)

fig, ax = plt.subplots(figsize=(7,6))

xs = np.arange(len(labels))
width = 1



labels = ["Invalid\nImage" if val == "image file is truncated (32 bytes not processed)" else "HTTP\n{0}".format(val) if val != "s3" else "S3\nGET" for val in labels]
rects = plt.bar(xs, values, width, align='center')
plt.xticks(xs, labels,  rotation='vertical')
plt.yticks(values)
plt.ylabel("Number Of Requests")

autolabel(rects)

display(fig)
fig.savefig("/dbfs/mnt/group07/plots/initial_downloads.pdf")

# COMMAND ----------

