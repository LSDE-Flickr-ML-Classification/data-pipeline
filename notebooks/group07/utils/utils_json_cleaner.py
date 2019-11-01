# Databricks notebook source


# COMMAND ----------

print(len(files))

# COMMAND ----------


zip_chunks = chunks(files, 500)

# COMMAND ----------



# COMMAND ----------

print(files)

# COMMAND ----------

import os
for file in files:
  os.remove("/dbfs/" + file.path[6:])
  print(file)

# COMMAND ----------

import os
os.remove("/dbfs/mnt/group07/final_data_product/visualization/json_output.tar")

# COMMAND ----------

import shutil
# shutil.rmtree("/dbfs/mnt/group07/final_data_product/visualization/json_output/", ignore_errors=False, onerror=None)

# COMMAND ----------

#files = dbutils.fs.ls("/mnt/group07/final_data_product/visualization/json_output")
#rdd = sc.parallelize(files)
#rdd.map(lambda p: os.remove("/dbfs/" + p.path[6:])).count()

# COMMAND ----------

