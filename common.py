import csv

import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import shutil
import itertools
from sklearn import metrics

LOCAL_MODE = True

if LOCAL_MODE:
    TRAIN_PATH = sys.argv[1]
    MODELS_FOLDER = sys.argv[2]

    assert os.path.isdir(TRAIN_PATH), "arg 1 must be a folder!"

    os.makedirs(MODELS_FOLDER, exist_ok=True)
    assert os.path.isdir(MODELS_FOLDER), "arg 2 must be a folder!"

# !mkdir -p /content/drive/Shareddrives/deep_learning/models

"""# ADDESTRAMENTO

## No augmentation
"""

BATCH_SIZE = 64
WIDTH, HEIGHT = 128, 65
EPOCHS = 50


# helper class to switch between color-modes
class Colors:
    class ColorMode:
        def __init__(self, keyword: str, channels: int):
            self.keyword = keyword
            self.channels = channels

    # Define color modes as class instances
    RGB = ColorMode('rgb', 3)
    GRAYSCALE = ColorMode('grayscale', 1)


COLOR_MODE = Colors.GRAYSCALE


# compile_model compiles a model
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


def get_model_folder(name: str):
    return os.path.join(MODELS_FOLDER, name)


def get_model_weights_path(name: str):
    return os.path.join(get_model_folder(name), "model.keras")


def plot(model: tf.keras.Model, history: any, X, Y, d: tf.data.Dataset, name: str):
    out_folder = get_model_folder(name)
    # Plot training history f1_score
    plt.figure(figsize=(3.66, 3.66))
    plt.plot(np.mean(history.history['f1_score'], axis=1), label='avg f1_score')
    plt.plot(np.mean(history.history['val_f1_score'], axis=1), label='avg val_f1_score')
    plt.xlabel('Epoch')
    plt.ylabel('F1-score')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()
    plt.draw()
    plt.savefig(os.path.join(out_folder, "learning_history-f1_score.png"), dpi=96*5)
    plt.close()

    # Plot training history loss
    plt.figure(figsize=(3.66, 3.66))
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower right')
    plt.show()
    plt.draw()
    plt.savefig(os.path.join(out_folder, "learning_history-loss.png"), dpi=96*5)
    plt.close()

    # Plot training history accuracy
    plt.figure(figsize=(3.66, 3.66))
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()
    plt.draw()
    plt.savefig(os.path.join(out_folder, "learning_history-acc.png"), dpi=96*5)
    plt.close()

    # PLot confusion matrix
    preds = model.predict(X)
    Y_pred = np.argmax(preds, axis=1)

    cm = metrics.confusion_matrix(Y, Y_pred, normalize='true')
    #cm = np.trunc(cm * 10 ** 2) / (10 ** 2)

    # LOG this should be equal to the original one
    #   correct_predictions = sum(1 for p, t in zip(Y_pred, Y_test) if p == t)
    #   total_predictions = len(Y_pred)
    #   accuracy = correct_predictions / total_predictions
    #   print("Accuracy: ", accuracy)
    # LOG

    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(N_CLASSES)
    plt.xticks(tick_marks, CLASSES, rotation=45)
    plt.yticks(tick_marks, CLASSES)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:.2f}".format(cm[i, j]), horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    plt.show()
    plt.draw()
    plt.savefig(os.path.join(out_folder, "confusion_matrix.png"), dpi=96*8)
    plt.close()

    # Save metrics on eval test on file
    res = model.evaluate(d, return_dict=True)
    with open(os.path.join(out_folder, "metrics.csv"), 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(res.keys()))
        writer.writeheader()
        writer.writerows([res])

    # F1-scores
    with open(os.path.join(out_folder, "f1.csv"), 'w') as file:
        file.write("class,f1\n")
        for i, c in enumerate(CLASSES):
            file.write(c + "," + "{:.3f}".format(res["f1_score"][i]) + "\n")


# train trains a model and put its weight in the specified output path
def train(model: any, name: str):
    # define useful callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        min_delta=0.02,
        patience=6,
    )

    save_best_model = tf.keras.callbacks.ModelCheckpoint(
        filepath=get_model_weights_path(name),
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
    )

    csv_logger = tf.keras.callbacks.CSVLogger(
        os.path.join(get_model_folder(name), "model.stats.csv"),
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

    return history


def evalutate(model: tf.keras.Model, name: str):
    model_out_folder = get_model_folder(name)

    if os.path.exists(model_out_folder):
        shutil.rmtree(model_out_folder)

    os.makedirs(model_out_folder, exist_ok=True)

    history = train(model, name)

    plot(model, history, X_val, Y_val, val_dataset, name)


"""La roba vera per l'addestrament insomma"""
# Load the dataset without validation splitting
dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    image_size=(HEIGHT, WIDTH),
    color_mode=COLOR_MODE.keyword,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True,
    seed=0xcafebabe
)

"""devide the dataset and log info"""

CLASSES = dataset.class_names
N_CLASSES = len(dataset.class_names)

# Calculate the number of validation samples
N_SAMPLES = dataset.cardinality().numpy()

VALIDATION_TEST_SAMPLES = int(0.35 * N_SAMPLES)  # 35% of data for validation and test
VALIDATION_SAMPLES = int(0.5 * VALIDATION_TEST_SAMPLES)

# Split the dataset into training and validation
val_test_dataset = dataset.take(VALIDATION_TEST_SAMPLES)
train_dataset = dataset.skip(VALIDATION_TEST_SAMPLES)

test_dataset = val_test_dataset.take(VALIDATION_SAMPLES)
val_dataset = val_test_dataset.skip(VALIDATION_SAMPLES)


def generate_eval_matrixes(dataset: tf.data.Dataset):
    X = []
    Y = []

    for images, labels in dataset:
        for image in images:
            X.append(np.array(image.numpy().tolist()))  # append list
        for label in labels:
            Y.append(np.argmax(label.numpy(), axis=0))

    X = tf.constant(X)
    return X, Y


X_val, Y_val = generate_eval_matrixes(val_dataset)
X_test, Y_test = generate_eval_matrixes(test_dataset)