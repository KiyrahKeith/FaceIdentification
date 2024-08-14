# Citation: Code for this file was copied from https://github.com/sazidabintaislam/Camera_trap_Journal_2022/
#           and modified for testing with my datasets.
import os
# import the needed packages
import matplotlib.pyplot as plt
import matplotlib.image as img
import tensorflow.keras as keras
import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from datetime import datetime
import itertools
from tensorflow.keras.callbacks import EarlyStopping
from keras.layers import Input
from keras.applications import VGG16
from keras.applications import ResNet50
from keras.models import Model
from keras.optimizers import SGD, Adam


def vgg16_1(num_classes):
    baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    # construct the head of the model that will be placed on top of the base model

    # Freeze four convolution blocks
    for layer in baseModel.layers:
        layer.trainable = False

    headModel = baseModel.output
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(512, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(256, activation="relu")(headModel)
    headModel = Dense(num_classes, activation="softmax")(headModel)
    # place the head FC model on top of the base model (this will become
    # the actual model we will train)
    model = Model(inputs=baseModel.input, outputs=headModel)

    opt = SGD(lr=1e-4, momentum=0.9)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["acc"])

    model.summary()

    # Make sure you have frozen the correct layers
    for i, layer in enumerate(model.layers):
        print(i, layer.name, layer.trainable)

    return model

def vgg16_2(num_classes):
    baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    # construct the head of the model that will be placed on top of the base model

    # Freeze four convolution blocks
    for layer in baseModel.layers[:15]:
        layer.trainable = False

    headModel = baseModel.output
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(256, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(256, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(num_classes, activation="softmax")(headModel)
    # place the head FC model on top of the base model (this will become
    # the actual model we will train)
    model = Model(inputs=baseModel.input, outputs=headModel)

    opt = SGD(lr=1e-4, momentum=0.9)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["acc"])

    return model

# This code is from file: multi_VGG16_3.ipynb
def vgg16_3(num_classes):
    baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    # construct the head of the model that will be placed on top of the
    # the base model

    # Freeze four convolution blocks
    for layer in baseModel.layers[:11]:
        layer.trainable = False

    headModel = baseModel.output
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(256, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(256, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(num_classes, activation="softmax")(headModel)
    # place the head FC model on top of the base model (this will become
    # the actual model we will train)
    model = Model(inputs=baseModel.input, outputs=headModel)

    opt = SGD(lr=1e-4, momentum=0.9)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["acc"])

    return model


# This code is from file: Resnet_multi_type1.ipynb
def resnet_type1(num_classes):
    baseModel = ResNet50(weights="imagenet", include_top=False,
                         input_tensor=Input(shape=(224, 224, 3)))

    # construct the head of the model that will be placed on top of the
    # the base model
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(256, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(num_classes, activation="softmax")(headModel)
    # place the head FC model on top of the base model (this will become
    # the actual model we will train)
    model = Model(inputs=baseModel.input, outputs=headModel)
    # loop over all layers in the base model and freeze them so they will
    # *not* be updated during the training process
    for layer in baseModel.layers:
        layer.trainable = False

    epochs = 100
    learning_rate = 1e-4
    opt = Adam(learning_rate=learning_rate, decay=learning_rate / epochs)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["acc"])

    return model