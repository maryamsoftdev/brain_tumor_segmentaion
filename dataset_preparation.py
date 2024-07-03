import os
import numpy as np
from custom_datagenerator import imageLoader
import splitfolders

# Directories and file lists
base_dir = r"C:\Users\computer house\Downloads\Segmentation of BraTS2020 Project\BraTS2020_TrainingData\input_data_128"
train_img_dir = os.path.join(base_dir, "train", "images")
train_mask_dir = os.path.join(base_dir, "train", "masks")
val_img_dir = os.path.join(base_dir, "val", "images")
val_mask_dir = os.path.join(base_dir, "val", "masks")

# Ensure directories exist
os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(train_mask_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(val_mask_dir, exist_ok=True)

# Split data into training and validation sets
splitfolders.ratio(base_dir, output=base_dir, seed=42, ratio=(.75, .25), group_prefix=None)

# List files in directories
train_img_list = os.listdir(train_img_dir)
train_mask_list = os.listdir(train_mask_dir)
val_img_list = os.listdir(val_img_dir)
val_mask_list = os.listdir(val_mask_dir)

# Data generator
batch_size = 2

train_img_datagen = imageLoader(train_img_dir, train_img_list,
                                train_mask_dir, train_mask_list, batch_size)
val_img_datagen = imageLoader(val_img_dir, val_img_list,
                              val_mask_dir, val_mask_list, batch_size)
