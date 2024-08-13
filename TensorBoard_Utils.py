import tensorflow as tf
from datetime import datetime
import io
import os
import shutil
import cv2
import itertools
from packaging import version

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics

dir = "logs"

def reset_directory():
    # If the current directory exists, delete it.
    try:
        # Check if the directory exists
        if os.path.exists(dir):
            # Use shutil.rmtree to recursively delete a directory and its contents
            shutil.rmtree(dir)
            print(f"Successfully deleted {dir}")
        else:
            print(f"{dir} does not exist")
    except Exception as e:
        print(f"Error deleting {dir}: {e}")

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def image_grid(name, dataset, class_names):
    """Return a 5x5 grid of the dataset as a matplotlib figure."""
    images, labels = next(iter(dataset))

    # Create a figure to contain the plot.
    figure = plt.figure(figsize=(9, 9))
    for i in range(25):
        # Start next subplot.
        plt.subplot(5, 5, i + 1, title=class_names[labels[i]])
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)

    logdir = dir + "/plots/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(logdir)

    # Convert to image and log
    with file_writer.as_default():
        tf.summary.image(name, plot_to_image(figure), step=0)


def normalize_images(image, label):
    # Normalize images float32 values in the range [0, 1]
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def sample_augmentations(image, name):
    # Create a figure to contain the plot.
    labels = [
        "Original",
        "Hue",
        "Hue",
        "Brightness",
        "Brightness",
        "Brightness",
        "Contrast",
        "Contrast",
        "Contrast",
        "Saturation",
        "Saturation",
        "Saturation",
        #"B+S",
        #"B+H",
        #"B+C"
    ]

    combo1 = tf.image.random_brightness(image, max_delta=0.5, seed=12345)
    combo1 = tf.image.random_saturation(combo1, lower=0.01, upper=0.1, seed=12345)

    combo2 = tf.image.random_brightness(image, max_delta=0.5, seed=12345)
    combo2 = tf.image.random_hue(combo2, max_delta=0.4, seed=12345)

    combo3 = tf.image.random_brightness(image, max_delta=0.5, seed=12345)
    combo3 = tf.image.random_contrast(combo3, lower=1.9, upper=2.0, seed=12345)

    rotate = keras.layers.RandomRotation(0.10)

    images = [
        image,
        tf.image.random_hue(image, max_delta=0.3),
        tf.image.random_hue(image, max_delta=0.5),
        tf.image.random_brightness(image, max_delta=0.15),
        tf.image.random_brightness(image, max_delta=0.3),
        tf.image.random_brightness(image, max_delta=0.5),
        tf.image.random_contrast(image, lower=0.2, upper=0.3),
        tf.image.random_contrast(image, lower=1.9, upper=2.0),
        tf.image.random_contrast(image, lower=2.5, upper=2.6),
        tf.image.random_saturation(image, lower=0.01, upper=0.1),
        tf.image.random_saturation(image, lower=0.2, upper=0.4),
        tf.image.random_saturation(image, lower=0.8, upper=0.9),
        #rotate(image),
        #rotate(image),
        #rotate(image)
    ]

    images = [tf.clip_by_value(img, 0.0, 1.0) for img in images]

    # Calculate the mean brightness value for each image, round to 3 decimal places, and append it onto that image's label.
    #for i, img in enumerate(images):
        # img_np = img.numpy()
        # mean_brightness = round(np.mean(img_np), 3)
        # labels[i] += " (" + str(mean_brightness) + ")" # Append the mean brightness label onto each image label

    figure = plt.figure(figsize=(9, 9))
    for i in range(12):
        # Start next subplot.
        plt.subplot(4, 3, i + 1, title=labels[i])
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)

    logdir = dir + "/plots/" + name + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(logdir)

    # Convert to image and log
    with file_writer.as_default():
        tf.summary.image("Sample Augmentations", plot_to_image(figure), step=0)


def augment(image, label):
    # Random brightness
    # image = tf.image.random_brightness(image, max_delta=0.05)

    # Random contrast
    # image = tf.image.random_contrast(image, lower=0.1, upper=0.2)

    # Random contrast
    image = tf.image.random_saturation(image, lower=0.1, upper=0.2)

    # Horizontal flip
    image = tf.image.random_flip_left_right(image)  # 50%

    return image, label


# Confusion Matrix implementation is used from Aladdin Persson
# (https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/TensorFlow/Basics/tutorial17-tensorboard)
def get_confusion_matrix(y_labels, logits, class_names):
    preds = np.argmax(logits, axis=1)
    cm = sklearn.metrics.confusion_matrix(
        y_labels, preds, labels=np.arange(len(class_names)),
    )

    return cm


def plot_confusion_matrix(cm, class_names):
    print("Plot confusion matrix")
    size = len(class_names)
    figure = plt.figure(figsize=(size, size))
    # plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.imshow(cm, interpolation="nearest", cmap=plt.colormaps.get_cmap('Blues'))
    plt.title("Confusion Matrix")

    indices = np.arange(len(class_names))
    plt.xticks(indices, class_names, rotation=45)
    plt.yticks(indices, class_names)

    # Normalize Confusion Matrix
    cm = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=3,)

    threshold = cm.max() / 2.0
    for i in range(size):
        for j in range(size):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(
                i, j, cm[i, j], horizontalalignment="center", color=color,
            )

    plt.tight_layout()
    plt.xlabel("True Label")
    plt.ylabel("Predicted label")

    cm_image = plot_to_image(figure)
    return cm_image
