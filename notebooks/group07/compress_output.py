# Databricks notebook source
import msgpack
import os
import json
import shutil

INPUT_FOLDER = "/dbfs/mnt/group07/final_data_product/visualization/json_output"
OUTPUT_FOLDER = "/dbfs/mnt/group07/final_data_product/visualization/json_output_compressed"
EXTENSION = ".json"
EXCLUDE_LIST = ['inverted_list.json', 'label_cloud.json', 'tag_list.json']


def get_json_as_object(json_file_location):
    json_file = open(json_file_location)
    json_raw = json_file.read()
    json_file.close()
    return json.loads(json_raw)


def write_bytes_to_file(bytes_content, output_file_location):
    bin_file = open(output_file_location, "wb")
    bin_file.write(bytes_content)
    bin_file.close()


def compress_and_store(file_location):
    json_obj = get_json_as_object(file_location) if EXTENSION == ".json" else None
    if json_obj is None:
        return
    msgpack_bytes = msgpack.packb(json_obj, use_bin_type=True)
    basename = os.path.splitext(os.path.basename(file_location))[0]
    full_output_file_path = OUTPUT_FOLDER + "/" + basename
    print("%s -> %s" % (file_location, full_output_file_path))
    write_bytes_to_file(msgpack_bytes, full_output_file_path)

    
def copy_metadata_to_folder(metadata_filename_list, location):
    for item in metadata_filename_list:
        full_path = INPUT_FOLDER + "/" + item
        print(full_path + "->" + location)
        shutil.copy(full_path, location)

    
all_file_paths = [os.path.join(INPUT_FOLDER, item) for item in os.listdir(INPUT_FOLDER) if item.endswith(EXTENSION) and item not in EXCLUDE_LIST]
copy_metadata_to_folder(EXCLUDE_LIST, OUTPUT_FOLDER)
sc.parallelize(all_file_paths).foreach(compress_and_store)


# COMMAND ----------

import tarfile
import os

OUTPUT_DIR = "/dbfs/mnt/group07/final_data_product/visualization/"
files = dbutils.fs.ls("mnt/group07/final_data_product/visualization/json_output_compressed")


dbutils.fs.rm(os.path.join(OUTPUT_DIR, "tar"), recurse=True)
dbutils.fs.mkdirs(os.path.join(OUTPUT_DIR, "tar"))

def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))
  
for index, chunk in enumerate(chunks(files, 1000)):
  print("creating tar from: {0}".format(index))
  tar = tarfile.open("/dbfs" + os.path.join(OUTPUT_DIR, "tar", "{0}.tar".format(index)), 'x:')
  for file in chunk:
    tar.add("/dbfs/" + file.path[6:])
  tar.close()

# COMMAND ----------

res = dbutils.fs.ls("dbfs:/mnt/group07/final_data_product/visualization/json_output_compressed")
total_size = 0
for item in res:
  total_size += item.size
print(total_size)