# Databricks notebook source
BUCKET_NUM = 7
INPUT_PARQUET = "/dbfs/mnt/group07/final_data_product/classification_result/unclassified_image_bytes.parquet"
OUTPUT_CLASSIFIED_PARQUET = "/dbfs/mnt/group07/final_data_product/classified_images/{0}/".format(BUCKET_NUM)
IMAGENET_CLASSES_LOCATION = "/dbfs/mnt/group07/imagenet_classes.json"

import concurrent.futures
import urllib.request
import os
import pandas
import io
import torch
import time
import json
import numpy as np

from PIL import Image
from fastparquet import ParquetFile
# from pyarrow.parquet import ParquetFile
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import models
from itertools import islice
import gc 

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import subprocess
import time 

# fxn taken from https://discuss.pytorch.org/t/memory-leaks-in-trans-conv/12492
def get_gpu_memory_map():   
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ])
    
    return float(result)

class ImageDataset(Dataset):

  MEAN = [0.485, 0.456, 0.406]
  STANDARD_DEVIATION = [0.229, 0.224, 0.225]

  image_normalize = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=MEAN, std=STANDARD_DEVIATION)
  ])

  def __init__(self, dataframe):
      self.tensors = ImageDataset.create_tensors_array(dataframe)

  def __len__(self):
      return len(self.tensors)

  def __getitem__(self, index):
      return self.tensors[index][0], self.tensors[index][1]
  
  @staticmethod
  def normalize_greyscale_image(image):
    normlize_greyscale =  transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,))
    ])
    
    try:
      normalized_greyscale_image = normlize_greyscale(image)
      rgb_normalized_greyscale_image = np.repeat(normalized_greyscale_image, 3, 0)
      return rgb_normalized_greyscale_image
    except Exception as e:
      print(e)
      return None
  
  @staticmethod
  def convert_image_bytes_to_image_tensor(image_bytes):
      image_bytearray = bytearray(image_bytes)
      image = Image.open(io.BytesIO(image_bytearray))
      try:
          return ImageDataset.image_normalize(image)
      except:
          return ImageDataset.normalize_greyscale_image(image)

  @staticmethod
  def create_tensors_array(dataframe):
      image_tensors = []
      for index, row in dataframe.iterrows():
          image_id = row['image_id']

          if row['img_binary'] is None:
              continue
          image_tensor = None
          try:
            image_tensor = ImageDataset.convert_image_bytes_to_image_tensor(row['img_binary'])
          except:
            pass
          if image_tensor is None:
              continue

          image_tensors.append((image_id, image_tensor))

      return image_tensors

def get_all_partitions_of_parquet(input_parquet):
    parquets_list = []
    for file in os.listdir(input_parquet):
        if file.endswith(".parquet"):
            parquets_list.append(os.path.join(input_parquet, file))
    print("Parquet file: %s has %d partitions!" % (input_parquet, len(parquets_list)))
    return parquets_list

def fetch_parquet_partitions(input_parquet):
    parquet_partitions = []
    parquet_partitions.extend(get_all_partitions_of_parquet(input_parquet))
    return parquet_partitions

import pyarrow.parquet as pq
def process_single_parquet_partition(parquet_location, output_folder):
  
    partion_file_name = os.path.basename(os.path.normpath(parquet_location))
    partion_id = partion_file_name[:10]
    
    if (os.path.isfile(os.path.join(output_folder, "{0}.csv".format(partion_id)))):
        # print("skipped existing output/ partid: {0}".format(partion_id))
        return None

    parquet_file = ParquetFile(parquet_location)
    
    print("Loading %s with %d row groups" % (parquet_location, len(parquet_file.row_groups)))        
    return parquet_file
  
def prepare_model_for_inference(compute_device):
    print("Initializing the classification model")
    model = models.inception_v3(pretrained=True)
    model.eval()
    model.to(compute_device)
    return model

def get_compute_device(can_use_cuda):
    is_cuda_usable = can_use_cuda and torch.cuda.is_available()
    compute_device = torch.device("cuda" if is_cuda_usable else "cpu")
    print("Using the following device for classification: %s" % compute_device)
    return compute_device
  
def get_classes(classes_file_location):
    if not os.path.exists(os.path.dirname(classes_file_location)):
        print("Could not find the classes file: %s" % classes_file_location)
        exit(-1)

    with open(classes_file_location) as json_file:
        classes_list = json.load(json_file)

    return [classes_list[str(k)][1] for k in range(len(classes_list))]
  
def get_matched_labels(predictions, image_ids, CLASSES):
    MAX_LABELS = 5
    results = pandas.DataFrame([])
    classfied_img_cnt = 0
    for index in range(0, len(predictions)):
        prediction = predictions[index]

        percentage = torch.nn.functional.softmax(prediction, dim=0) * 100
        _, indices = torch.sort(prediction, descending=True)

        matched_classes = [{"id": image_ids[index], "label": CLASSES[idx], "confidence": percentage[idx].item()} for idx in islice(indices, MAX_LABELS)]

        results = results.append(matched_classes)
        classfied_img_cnt += 1
    return (classfied_img_cnt ,results)
  
def classify_images(pd_df, model, CLASSES, DEVICE):
    start_time_classification = time.time()
    
    with torch.no_grad():
      tensors_dataset = ImageDataset(pd_df)
      # print("Classifing RG of %d images" % len(tensors_dataset))
      
      loader = torch.utils.data.DataLoader(tensors_dataset, batch_size=64, num_workers=0)
      results = pandas.DataFrame([])
      
      images_classified = 0
      for img_batch in loader:
          image_ids = img_batch[0]
          image_tensors = img_batch[1]
          predictions = model(image_tensors.to(DEVICE)).cpu()
          labelling_result = get_matched_labels(predictions, image_ids, CLASSES)
          results = results.append(labelling_result[1])
          images_classified += labelling_result[0]
      
      loader = None
      
      duration_classification = time.time() - start_time_classification
      speed = images_classified / duration_classification
      print("Classified RG - {0} imgs in: {1:.2f}, thats: {2:.2f} img/sec".format(images_classified, duration_classification, speed))
        
    return (images_classified, results)

def process_partition(partition, output_folder, model, CLASSES, DEVICE):
  start_time_classification = time.time()
  parquet_file = process_single_parquet_partition(partition, output_folder)
  
  if parquet_file is None:
    return (partition, None)
  partion_id = os.path.basename(os.path.normpath(partition))[:10]
  print("Processing P {0}".format(partion_id))
  
  results = pandas.DataFrame([])
  images_classified = 0
  
  for df in parquet_file.iter_row_groups(columns=["image_id", "img_binary", "download_link"]):
    result = classify_images(df, model, CLASSES, DEVICE)
    results = results.append(result[1])
    images_classified += result[0]
    
  duration_classification = time.time() - start_time_classification
  speed = images_classified / duration_classification
  print("Finished P {0} with {1} imags in: {2:.2f}, thats: {3:.2f} img/sec".format(partion_id, images_classified, duration_classification, speed))
  torch.cuda.empty_cache()
  return (partition, results)
  



PARQUE_PARTITIONS = fetch_parquet_partitions(INPUT_PARQUET)
DEVICE = get_compute_device(True)
MODEL = prepare_model_for_inference(DEVICE)
CLASSES = get_classes(IMAGENET_CLASSES_LOCATION)

os.makedirs (OUTPUT_CLASSIFIED_PARQUET, exist_ok=True)

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    # Load 2 Parquet files in parallel
    futures = { executor.submit(process_partition, prt, OUTPUT_CLASSIFIED_PARQUET, MODEL, CLASSES, DEVICE): prt for prt in PARQUE_PARTITIONS }
    for future in concurrent.futures.as_completed(futures):
        partition_result = futures[future]
        try:
          data = future.result()
        except Exception as exc:
            print('%r generated an exception: %s' % (partition_result, exc))
        else:
            partion_id = os.path.basename(os.path.normpath(data[0]))[:10]
            if data[1] is None:
                pass
                # print("Partition {0} already processed".format(partion_id))
            else: 
                output_filepath = os.path.join(OUTPUT_CLASSIFIED_PARQUET, "{0}.csv".format(partion_id))
                data[1].to_csv(output_filepath, sep=',',index=False)
                print("Finished and written partition {0}".format(partion_id))
    print("finished file 🥳")
    dbutils.library.restartPython() # Clear GPU memory
    print("gpu mem {0}".format(get_gpu_memory_map()))