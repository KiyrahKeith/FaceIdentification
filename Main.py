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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

utils.reset_directory()  # Clear the previous TensorBoard log directory

# Initialize parameters ###############################################################################################
img_size = 224  # The width and height dimensions of the image
batch_size = 32
epochs = 10
num_classes = 24  # There are 24 individual chimps in the C-Zoo dataset
validation_split = 0.1
optimizer = keras.optimizers.legacy.Adam(learning_rate=0.001)
loss = keras.losses.SparseCategoricalCrossentropy()
k_fold = 5  # The number of folds for the k-fold cross validation


# Running parameters (parameters that determine which pieces of the program will be completed during each run)
# An array of boolean values that indicate which elements of TensorBoard will be computed during the current run.
myTensorBoard = [0,  # Image augmentation grid
                 0,  # Sample image grid from training set
                 0,  # Confusion matrix
                 ]

# 0 - Don't train, 1 - Train with normal split, 2 - Train with k-fold cross validation
willTrain = 1  # willTrain is True if the model will train and evaluate the CNN
willAugment = False  # willAugment is True if the training set will be augmented during training.
# #####################################################################################################################

def preprocess_data(X, Y):
    X_p = keras.applications.resnet50.preprocess_input(X)
    Y_p = keras.utils.to_categorical(Y, num_classes)
    return X_p, Y_p


# Load dataset from directory
if willTrain == 1:
    # For normal training, load the training data WITH the validation set
    train_ds, validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
        'CZoo/Chimpanzee/',
        labels='inferred',
        label_mode="int",
        color_mode='rgb',
        batch_size=batch_size,
        image_size=(img_size, img_size),
        shuffle=True,
        seed=123,
        validation_split=0.1,
        subset="both",
    )

if willTrain == 2:
    # For k-fold cross validation, load the training data WITHOUT the validation set
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        'Dataset/',
        labels='inferred',
        label_mode="int",
        color_mode='rgb',
        batch_size=batch_size,
        image_size=(img_size, img_size),
        shuffle=True,
        seed=123
    )

# Get the list of class names
class_names = train_ds.class_names
# images, labels = next(iter(train_ds))
train_ds = train_ds.map(utils.normalize_images)
validation_ds = validation_ds.map(utils.normalize_images)

# Store the datasets in cache to reduce loading time.
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Configure model ########################################################################
# Create model layers
if willAugment:
    train_ds = train_ds.map(utils.augment)  # Augment the training set


# Normal training
if willTrain == 1:
    # Use callbacks to plot accuracy and loss into TensorBoard
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=utils.dir, histogram_freq=1,
    )

    x_train, y_train = next(iter(train_ds))
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = next(iter(validation_ds))
    x_test, y_test = preprocess_data(x_test, y_test)
    # print((x_train.shape, y_train.shape))

    input_t = keras.Input(shape=(img_size, img_size, 3))
    resnet = tf.keras.applications.ResNet50(include_top=False,
                                            input_tensor=input_t,
                                            weights='imagenet')

    for layer in resnet.layers:  #:170
        layer.trainable = False
    
    # Check the freezed was done ok
    '''for i, layer in enumerate(res_model.layers):
        print(i, layer.name, "-", layer.trainable)'''

    model = Sequential()
    model.add(layers.Lambda(lambda image: tf.image.resize(image, (img_size, img_size))))
    model.add(resnet)
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
                  optimizer=keras.optimizers.legacy.Adam(),
                  metrics=['accuracy'])

    history = model.fit(train_ds, batch_size=batch_size, epochs=epochs, verbose=2,
                        validation_data=validation_ds,
                        callbacks=[tensorboard_callback])

    # Create the confusion matrix in TensorBoard
    if myTensorBoard[2]:
        loss_fn = loss
        optimizer = keras.optimizers.legacy.Adam(learning_rate=0.001)
        acc_metric = keras.metrics.SparseCategoricalAccuracy()
        train_step = test_step = 0
        train_writer = tf.summary.create_file_writer("logs/train/")

        for epoch in range(epochs):
            confusion = np.zeros((len(class_names), len(class_names)))

            # Iterate through training set
            for batch_idx, (x, y) in enumerate(train_ds):
                with tf.GradientTape() as tape:
                    y_pred = model(x, training=True)
                    loss = loss_fn(y, y_pred)

                gradients = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(gradients, model.trainable_weights))
                acc_metric.update_state(y, y_pred)
                confusion += utils.get_confusion_matrix(y, y_pred, class_names)

            with train_writer.as_default():
                tf.summary.image(
                    "Confusion Matrix",
                    utils.plot_confusion_matrix(confusion / batch_idx, class_names),
                    step=epoch,
                )

            # Reset accuracy in between epochs (and for testing and test)
            acc_metric.reset_states()

# K-Fold Cross-Validation ########################################################################
if willTrain == 2:
    X = []
    Y = []
    for images, labels in train_ds:
        X.append(images.numpy())
        Y.append(labels.numpy())

    X = np.concatenate(X)
    Y = np.concatenate(Y)
    kf = KFold(k_fold, shuffle=True, random_state=42)
    oos_y = []
    oos_pred = []

    fold = 0
    for train, test in kf.split(X, Y):
        fold += 1
        print(f"Fold #{fold}")
        # images, labels = next(iter(train_ds))

        x_train = X[train]
        y_train = Y[train]
        x_test = X[test]
        y_test = Y[test]

        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=utils.dir, histogram_freq=1,
        )

        model.fit(x_train,
                  y_train,
                  epochs=epochs,
                  batch_size=batch_size,
                  validation_data=(x_test, y_test),
                  callbacks=[tensorboard_callback],
                  verbose=2,
                  )

if myTensorBoard[0]:  # Only create this TensorBoard section if it is activated at the top of this file.
    images, labels = next(iter(train_ds))
    for i in range(5):
        utils.sample_augmentations(images[i], str(i))

if myTensorBoard[1]:  # Create a sample grid of the training set images and load it into TensorBoard
    utils.image_grid("Training Set", train_ds, class_names)


# Default model.fit parameters
'''
model.fit(train_ds,
                  epochs=epochs,
                  validation_data=validation_ds,
                  callbacks=[tensorboard_callback],
                  verbose=2,
                  )'''

'''
# Normalize the dataset to be in the range 0-1
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
'''









