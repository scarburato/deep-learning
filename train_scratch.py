# -*- coding: utf-8 -*-
"""train_scratch.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/149X0IBt7SwsCtlznok8OAKAqr8GQU9Cz

# Train-Scratch

In this file we TRAIN.
"""

LOCAL_MODE = False

#if not LOCAL_MODE:
#  from google.colab import drive
#  from google.colab.patches import cv2_imshow
#
#  drive.mount('/content/drive', force_remount=True)
#
#  !cd /content; rm -r processed; 7z x drive/Shareddrives/deep_learning/processed.128_87.7z; ls -alF processed
#
#!pip install tensorflow --quiet

import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt

#!mkdir -p /content/drive/Shareddrives/deep_learning/models

"""# ADDESTRAMENTO

## No augmentation
"""

BATCH_SIZE = 32
WIDTH, HEIGHT = 128, 87
TRAIN_PATH = "/content/processed/"
LOCAL_MODELS_FOLDER = "/content/drive/Shareddrives/deep_learning/models"
EPOCHS = 30

# helper class to switch between color-modes
class Colors:
    class ColorMode:
        def __init__(self, keyword:str, channels:int):
            self.keyword = keyword
            self.channels = channels

    # Define color modes as class instances
    RGB = ColorMode('rgb', 3)
    GRAYSCALE = ColorMode('grayscale', 1)


COLOR_MODE = Colors.GRAYSCALE

def compile_model(model):
  model.compile(
      optimizer='adam',
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'],
  )

def trainmaxx(model, name):
    # define useful callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        min_delta=0.05,
        patience=6,
    )

    save_best_model = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(LOCAL_MODELS_FOLDER, name),
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
    )

    # csv logger
    logpath = os.path.join(LOCAL_MODELS_FOLDER, name + "_stats.csv")
    if os.path.exists(logpath):
      os.remove(logpath)

    csv_logger = tf.keras.callbacks.CSVLogger(
      logpath,
      append=True
    )

    # Train the model
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,

        callbacks=[
            csv_logger,
            early_stop,
            save_best_model
        ],

        verbose=1,
        workers=4
    )


    # Plot training history (optional)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()

try:
  resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
  tf.config.experimental_connect_to_cluster(resolver)
  tf.tpu.experimental.initialize_tpu_system(resolver)
  print("##### All devices: ", tf.config.list_logical_devices('TPU'))

  strategy = tf.distribute.TPUStrategy(resolver)
except ValueError:
  print("##### TPU not found using default strategy #####")
  strategy = tf.distribute.get_strategy()

"""La roba vera per l'addestrament insomma"""

from tensorflow.keras.utils import image_dataset_from_directory

# Load the dataset without validation splitting
dataset = image_dataset_from_directory(
    TRAIN_PATH,
    image_size=(HEIGHT, WIDTH),
    color_mode=COLOR_MODE.keyword,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
)

N_CLASSES = len(dataset.class_names)

# Calculate the number of validation samples
N_SAMPLES = dataset.cardinality().numpy()

VALIDATION_SAMPLES = int(0.2 * N_SAMPLES)  # 20% of data for validation

# Split the dataset into training and validation
train_dataset = dataset.skip(VALIDATION_SAMPLES)
val_dataset = dataset.take(VALIDATION_SAMPLES)

def CreateModel():
  model = tf.keras.Sequential([
      tf.keras.Input(shape=(HEIGHT, WIDTH, COLOR_MODE.channels)),
      tf.keras.layers.Rescaling(1./255),
      tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
      tf.keras.layers.MaxPooling2D((2, 2)),
      tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
      tf.keras.layers.MaxPooling2D((2, 2)),
      tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
      tf.keras.layers.MaxPooling2D((2, 2)),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.Dense(256, activation='relu'),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.Dense(N_CLASSES, activation='softmax')
  ])

  compile_model(model)

  return model

with strategy.scope():
  model = CreateModel()

model.summary()

trainmaxx(model, "model0")

"""## Augmentation


"""

def CreateModel():

  data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"), # Applies horizontal flipping to a random 50% of the images
    tf.keras.layers.RandomRotation(0.1), # Rotates the input images by a random value in the range[–10%, +10%] (fraction of full circle [-36°, 36°])
    tf.keras.layers.RandomZoom(0.2), # Zooms in or out of the image by a random factor in the range [-20%, +20%]
  ], name="ruotaingrandimento")

  model = tf.keras.Sequential([
      tf.keras.Input(shape=(HEIGHT, WIDTH, COLOR_MODE.channels)),
      tf.keras.layers.Rescaling(1./255),
      #data_augmentation,
      tf.keras.layers.RandomContrast(0.5),
      tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu",),
      tf.keras.layers.MaxPooling2D(pool_size=2),

      tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu"),
      tf.keras.layers.MaxPooling2D(pool_size=2),

      tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu"),
      tf.keras.layers.MaxPooling2D(pool_size=2),

      tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation="relu"),
      tf.keras.layers.MaxPooling2D(pool_size=2),

      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Dense(N_CLASSES, activation='softmax')
  ])


  # Compile the model
  compile_model(model)

  return model

# TODO: check if creating the model inside this scope is correct
with strategy.scope():
  model = CreateModel()

model.summary()

trainmaxx(model, "model1")

def CreateModel():
  model = tf.keras.Sequential([
      tf.keras.Input(shape=(HEIGHT, WIDTH, COLOR_MODE.channels)),
      tf.keras.layers.Rescaling(1./255),
      tf.keras.layers.Conv2D(512, (3, 3), activation='relu6'),
      tf.keras.layers.MaxPooling2D((2, 2)),
      tf.keras.layers.Conv2D(1024, (3, 3), activation='relu'),
      tf.keras.layers.MaxPooling2D((2, 2)),
      tf.keras.layers.Conv2D(2048, (3, 3), activation='relu'),
      tf.keras.layers.MaxPooling2D((2, 2)),
      tf.keras.layers.Conv2D(1024, (3, 3), activation='relu'),
      tf.keras.layers.MaxPooling2D((2, 2)),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(512, activation='relu'),
      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.Dense(512, activation='relu'),
      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.Dense(N_CLASSES, activation='softmax')
  ])

  compile_model(model)

  return model

with strategy.scope():
  model = CreateModel()

model.summary()

trainmaxx(model, "model2")