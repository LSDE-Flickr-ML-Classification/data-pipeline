# Databricks notebook source
files = dbutils.fs.ls("/mnt/group07/final_data_product/visualization/json_output")

# COMMAND ----------

import os
bytessize = 0
for file in files:
  bytessize += files.size
print(bytessize)

# COMMAND ----------

