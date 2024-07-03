import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from custom_datagenerator import imageLoader
from simple_3d_Unet import simple_unet_model
import segmentation_models_3D as sm

# Check GPU availability and set memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# Directories and file lists
base_dir = r"C:\Users\computer house\Downloads\Segmentation of BraTS2020 Project\BraTS2020_TrainingData\input_data_128"
train_img_dir = os.path.join(base_dir, "train", "images")
train_mask_dir = os.path.join(base_dir, "train", "masks")
val_img_dir = os.path.join(base_dir, "val", "images")
val_mask_dir = os.path.join(base_dir, "val", "masks")

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

# Model and training parameters
model = simple_unet_model(IMG_HEIGHT=128, IMG_WIDTH=128, IMG_DEPTH=128, IMG_CHANNELS=3, num_classes=4)

LR = 0.0001
optim = Adam(LR)

# Define custom IOU metric
def iou_score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return sm.metrics.IOUScore(threshold=0.5)(y_true, y_pred)

model.compile(optimizer=optim, loss=sm.losses.DiceLoss(), metrics=[iou_score])

# Training
steps_per_epoch = len(train_img_list) // batch_size
val_steps_per_epoch = len(val_img_list) // batch_size

history = model.fit(train_img_datagen,
                    steps_per_epoch=steps_per_epoch,
                    epochs=20,
                    verbose=1,
                    validation_data=val_img_datagen,
                    validation_steps=val_steps_per_epoch)

model.save('brats_3d.hdf5')

# Plotting training and validation metrics
def plot_metrics(history, metric_name):
    plt.plot(history.history[metric_name], 'y', label=f'Training {metric_name}')
    plt.plot(history.history[f'val_{metric_name}'], 'r', label=f'Validation {metric_name}')
    plt.title(f'Training and validation {metric_name}')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    plt.show()

plot_metrics(history, 'loss')
plot_metrics(history, 'iou_score')

# Load model and make predictions
my_model = load_model('brats_3d.hdf5', custom_objects={'iou_score': iou_score})

# Predictions and visualization
test_image_batch, test_mask_batch = val_img_datagen.__next__()
test_pred_batch = my_model.predict(test_image_batch)

n_slice = 55
plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_image_batch[0][:, :, n_slice, 1], cmap='gray')
plt.subplot(232)
plt.title('Prediction on test image')
plt.imshow(np.argmax(test_pred_batch[0], axis=-1)[:, :, n_slice])
plt.show()
