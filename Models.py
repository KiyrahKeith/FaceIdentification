import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.model_selection import KFold
import TensorBoard_Utils as utils
import Callbacks
from keras.applications import (VGG16, ResNet50, InceptionV3, InceptionResNetV2, MobileNetV2, DenseNet201, Xception)
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from keras.applications.resnet import preprocess_input as resnet_preprocess
from keras.applications.inception_v3 import preprocess_input as inceptionv3_preprocess
from keras.applications.inception_resnet_v2 import preprocess_input as inceptionresnestv2_preprocess
from keras.applications.mobilenet_v2 import preprocess_input as mobilenetv2_preprocess
from keras.applications.densenet import preprocess_input as densenet_preprocess
from keras.applications.xception import preprocess_input as xception_preprocess


# Stores the information for each transfer model available
# Key = model name, Value = (Model loading name, image size, dataset preprocessing function)
models_dict = {
    "VGG16": (VGG16, 224, vgg16_preprocess),
    "ResNet50": (ResNet50, 224, resnet_preprocess),
    "InceptionV3": (InceptionV3, 224, inceptionv3_preprocess),
    "InceptionResNetV2": (InceptionResNetV2, 224, inceptionresnestv2_preprocess),
    "MobileNetV2": (MobileNetV2, 224, mobilenetv2_preprocess),
    "DenseNet201": (DenseNet201, 224, densenet_preprocess),
    "Xception": (Xception, 299, xception_preprocess)
}


# Load dataset from a directory and split into training and validation data
def load_data(directory, batch_size, validation_split):
    train_ds, validation_ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels='inferred',
        label_mode="int",
        color_mode='rgb',
        batch_size=batch_size,
        shuffle=True,
        seed=123,
        validation_split=validation_split,
        subset="both",
    )

    train_ds = train_ds.map(utils.normalize_images)
    validation_ds = validation_ds.map(utils.normalize_images)

    print("Data load successful")
    return train_ds, validation_ds


# Load the requested model from the models_dict
def get_transfer_model(model_name):
    if model_name in models_dict:
        model_class, img_size, preprocess = models_dict[model_name]
        input_t = keras.Input(shape=(img_size, img_size, 3))
        transfer_model = model_class(include_top=False,
                                     input_tensor=input_t,
                                     weights='imagenet')

        for layer in transfer_model.layers:
            layer.trainable = False

        print("Transfer model load successful")
        return transfer_model
    else:
        raise ValueError(f"Model '{model_name}' is not recognized. Please choose from {list(models_dict.keys())}.")


# Preprocess the training and validation set based on the transfer-learning model used
def preprocess_data(model_name, train_ds, validation_ds):
    if model_name in models_dict:
        model_class, img_size, preprocess = models_dict[model_name]
        # Apply preprocessing to both the training and validation datasets
        train_ds = train_ds.map(lambda img, lbl: preprocess_data_util(img, lbl, preprocess, img_size),
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        validation_ds = validation_ds.map(lambda img, lbl: preprocess_data_util(img, lbl, preprocess, img_size),
                                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Optimize data pipeline with prefetching
        train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        validation_ds = validation_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        print("Preprocessing successful")
        return train_ds, validation_ds
    else:
        raise ValueError(f"Model '{model_name}' is not recognized. Please choose from {list(models_dict.keys())}.")


def preprocess_data_util(image, label, preprocess, img_size):
    image = tf.image.resize(image, [img_size, img_size])
    image = preprocess(image)
    return image, label


# Returns a new transfer-learning model
def create_model(model_name, learning_rate, num_classes, optimizer, metric):
    # This architecture was obtained from
    # https://hamdi-ghorbel78.medium.com/a-guide-to-transfer-learning-with-keras-using-resnet50-453934a7b7dc
    model = Sequential()
    model.add(get_transfer_model(model_name))
    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=metric)
    return model


def train_model(model, train_ds, validation_ds, batch_size, epochs, verbose, record_name, record_data):
    # Prepare the csv_logger to save model metrics to a file
    csv_logger = Callbacks.CustomCSVLogger('results.csv',
                                       append=True,
                                       separator=",",
                                       record_name=record_name,
                                       record_data=record_data)

    log_dir = os.path.join(utils.dir, record_name)
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1,
    )

    return model.fit(train_ds,
                     batch_size=batch_size,
                     epochs=epochs,
                     verbose=verbose,
                     validation_data=validation_ds,
                     callbacks=[csv_logger, tensorboard_callback])



