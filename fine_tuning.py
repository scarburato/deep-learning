# -*- coding: utf-8 -*-
"""fine_tuning.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1o0IEBjNgJlLE6j93eZKgojF2H9dhGAzF

## Feature extraction
"""

LOCAL_MODE = True

# if not LOCAL_MODE:
#   from google.colab import drive
#   from google.colab.patches import cv2_imshow
#
#   drive.mount('/content/drive', force_remount=True)
#   !cd /content; rm -r processed; 7z x drive/Shareddrives/deep_learning/processed.128_65.7z; ls -alF processed
#
# !pip install tensorflow --quiet

import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import sys


if LOCAL_MODE:
  TRAIN_PATH = sys.argv[1]
  LOCAL_MODELS_FOLDER = sys.argv[2]

  assert os.path.isdir(TRAIN_PATH), "arg 1 must be a folder!"

  os.makedirs(LOCAL_MODELS_FOLDER, exist_ok=True)
  assert os.path.isdir(LOCAL_MODELS_FOLDER), "arg 2 must be a folder!"

BATCH_SIZE = 64
WIDTH, HEIGHT = 128, 65
EPOCHS = 25


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

  import tensorflow_addons as tfa

  f1 = tfa.metrics.F1Score(num_classes=N_CLASSES)

  model.compile(
      optimizer='adam',
      loss='categorical_crossentropy',
      metrics=[
        f1,
        "accuracy"
      ],
  )


def trainmaxx(model, name):
    out_folder = os.path.join(LOCAL_MODELS_FOLDER, name)
    if os.path.exists(out_folder):
      shutil.rmtree(out_folder)

    os.makedirs(out_folder, exist_ok=True)

    # define useful callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        min_delta=0.05,
        patience=6,
    )

    save_best_model = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(out_folder, name + ".h5"),
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
    )

    # csv logger
    logpath = os.path.join(out_folder, "stats.csv")
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
        validation_steps=len(val_dataset),
        callbacks=[
            csv_logger,
            early_stop,
            save_best_model
        ],

        verbose=1,
        workers=4
    )

    # Plot training history (optional)
    plt.figure(figsize=(10,10))
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()
    plt.savefig(os.path.join(out_folder, "learning_history.png"))
    plt.close()

    # PLot confusion graph
    preds = model.predict(test_dataset)
    Y_pred = np.argmax(preds, axis=1)

    # Confusion matrix

    rounded_labels=np.argmax(Y_test, axis=1)
    rounded_labels[1]

    cm = metrics.confusion_matrix(rounded_labels, Y_pred, normalize = 'true')
    cm = np.trunc(cm*10**2)/(10**2)

    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(N_CLASSES)
    plt.xticks(tick_marks, CLASSES, rotation=45)
    plt.yticks(tick_marks, CLASSES)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    plt.show()
    plt.savefig(os.path.join(out_folder, "confusion_matrix.png"))
    plt.close()

from tensorflow.keras.utils import image_dataset_from_directory

# Load the dataset without validation splitting
dataset = image_dataset_from_directory(
    TRAIN_PATH,
    image_size=(HEIGHT, WIDTH),
    color_mode= COLOR_MODE.keyword,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=0xcafebabe,
)

CLASSES = dataset.class_names
N_CLASSES = len(dataset.class_names)

# Calculate the number of validation samples
N_SAMPLES = dataset.cardinality().numpy()

VALIDATION_SAMPLES = int(0.175 * N_SAMPLES)  # 20% of data for validation

# Split the dataset into training and validation
train_dataset = dataset.skip(VALIDATION_SAMPLES)
val_dataset = dataset.take(VALIDATION_SAMPLES)
train_dataset = train_dataset.skip(VALIDATION_SAMPLES)
test_dataset = train_dataset.take(VALIDATION_SAMPLES)

X_test = []
Y_test = []

for images, labels in test_dataset:
    for image in images:
        X_test.append(image)                    # append tensor
        #X.append(image.numpy())           # append numpy.array
        #X.append(image.numpy().tolist())  # append list
    for label in labels:
        Y_test.append(label)                    # append tensor
        #Y.append(label.numpy())           # append numpy.array
        #Y.append(label.numpy().tolist())  # append list

"""Let's use the convolutional base of the VGG16 network, trained on ImageNet, to extract interesting features from
our audio images.
"""

from tensorflow.keras.applications import vgg16

conv_base = vgg16.VGG16(
    input_shape=(HEIGHT,WIDTH,3),
    weights="imagenet",
    include_top=False,
)

conv_base.summary()

"""## Layers freezing
Before we compile and train our model, a very important thing to do is to freeze the convolutional base. "Freezing" a layer or set of
layers means preventing their weights from getting updated during training. If we don't do this, then the representations that were
previously learned by the convolutional base would get modified during training. Since the classifier on top (i.e., the `Dense` layers we will add) is randomly initialized,
very large weight updates would be propagated through the network, effectively destroying the representations previously learned.

In Keras, freezing a network is done by setting its `trainable` attribute to `False`:
"""

import numpy as np

print('This is the number of trainable weights '
      'before freezing the conv base:', sum(np.prod(x.shape) for x in conv_base.trainable_weights))

conv_base.trainable = False

print('This is the number of trainable weights '
      'after freezing the conv base:', sum(np.prod(x.shape) for x in conv_base.trainable_weights))

"""Now we can create a new model that chains together
1. A data augmentation stage
2. Our frozen convolutional base
3. A dense classifier
"""

class Gray2VGGInput(tf.keras.layers.Layer):
    """
    Custom conversion layer
    """
    def build( self, x ) :
        self.image_mean = tf.keras.backend.variable(
            value=np.array([103.939, 116.779, 123.68])
              .reshape([1,1,1,3]).astype('float32'), dtype='float32', name='imageNet_mean' )
        self.built = True
        return

    def call( self, x ) :

        rgb_x = tf.keras.backend.concatenate( [x,x,x], axis=-1 )
        norm_x = rgb_x - self.image_mean
        return norm_x

    def compute_output_shape( self, input_shape ) :
        return input_shape[:3] + (3,)


model = tf.keras.Sequential([
    tf.keras.Input(shape=(HEIGHT, WIDTH, COLOR_MODE.channels), name="ingresso-monocromo"),
    # Add remove this if colormode is rgb
    Gray2VGGInput(name = "convertitore-a-colori"),
    tf.keras.layers.Rescaling(1./255),
    conv_base,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(2048, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2048, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(2038, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(N_CLASSES, activation='softmax'),
], name="layer-freeze-VGG16")

model.compile(
    #loss="categorical_crossentropy",
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)

model.summary()

trainmaxx(model, "vgg16_feature_extract")

"""# Fine tuning

"""

conv_base.summary()

conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

conv_base.summary()

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)

model.summary()

trainmaxx(model, "vgg16_finetune")