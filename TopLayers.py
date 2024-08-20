import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential


def get_top_layers(architecture, num_classes):
    print(f"Load architecture {architecture}")
    model = Sequential()
    if architecture == 0:  # Equal dense layers
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(256, activation='relu'))
        model.add(keras.layers.Dense(256, activation='relu'))
    elif architecture == 1:  # Descending dense layers
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(256, activation='relu'))
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dense(64, activation='relu'))
    elif architecture == 2:  # Descending dense layers with batch norm
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(256, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(64, activation='relu'))
    elif architecture == 3:  # Descending dense layers with batch norm and dropout
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(256, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.BatchNormalization())
    elif architecture == 4:  # Default but with no batch norm
        model.add(keras.layers.GlobalAveragePooling2D())
        model.add(keras.layers.Dense(256, activation='relu'))
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dropout(0.3))
    elif architecture == 5:  # Default but with no Dropout
        model.add(keras.layers.GlobalAveragePooling2D())
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(256, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.BatchNormalization())
    elif architecture == 6:  # Default architecture but with dropout at 0.5
        model.add(keras.layers.GlobalAveragePooling2D())
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(256, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.BatchNormalization())
    else: # Default architecture from my original tests
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
    return model

