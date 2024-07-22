import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.model_selection import KFold
import TensorBoard_Utils as utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

utils.reset_directory()  # Clear the previous TensorBoard log directory

# Initialize parameters ###############################################################################################
img_size = 128  # The width and height dimensions of the image
batch_size = 32
epochs = 5
num_classes = 24  # There are 24 individual chimps in the C-Zoo dataset
k_fold = 5  # The number of folds for the k-fold cross validation

# Running parameters (parameters that determine which pieces of the program will be completed during each run)

# An array of boolean values that indicate which elements of TensorBoard will be computed during the current run.
myTensorBoard = [1,  # Image Augmentation Grid
                 ]

willTrain = True  # willTrain is True if the model will train and evaluate the CNN
willAugment = False  #willAugment is True if the training set will be augmented during training.
# #####################################################################################################################


# Load dataset from directory
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'Dataset/',
    labels='inferred',
    label_mode="int",
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(img_size, img_size),
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset="training",
)

validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'Dataset/',
    labels='inferred',
    label_mode="int",
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(img_size, img_size),
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset="validation",
)

# Get the list of class names (if needed)
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
model = keras.Sequential([
    layers.Input((img_size, img_size, 3)),
    layers.Conv2D(16, 3, padding='same'),
    layers.Conv2D(32, 3, padding='same'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(num_classes),
])

# Set training settings
model.compile(
    optimizer=keras.optimizers.legacy.Adam(),
    loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True)],
    metrics=["accuracy"],
)


# K-Fold Cross-Validation ########################################################################
if willTrain:
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
        images, labels = next(iter(train_ds))

        x_train = images[train]
        y_train = labels[train]
        x_test = images[test]
        y_test = labels[test]

        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=utils.dir, histogram_freq=1,
        )

        model.fit((x_train, y_train),
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  callbacks=[tensorboard_callback],
                  verbose=2,
                  )

if willAugment:
    train_ds = train_ds.map(utils.augment)  # Augment the training set

# Create a sample grid of the training set images and load it into TensorBoard
# utils.image_grid("Training Set", train_ds, class_names)



if myTensorBoard[0]:  # Only create this TensorBoard section if it is activated at the top of this file.
    images, labels = next(iter(train_ds))
    for i in range(5):
        utils.sample_augmentations(images[i], str(i))

# Default model.fit parameters
'''
model.fit(train_ds,
                  epochs=epochs,
                  validation_data=validation_ds,
                  callbacks=[tensorboard_callback],
                  verbose=2,
                  )'''

'''
# Show a sample of one of the images
for images, labels in train_ds.take(1):
    # Choose the first image from the batch
    image = images[0].numpy().astype("uint8")  # Convert image tensor to numpy array

    # Display the image
    plt.imshow(image)
    plt.title(f"Label: {labels[0].numpy()}")
    plt.axis("off")
    plt.show()

# Normalize the dataset to be in the range 0-1
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

def augment(x, y):
    image = tf.image.random_brightness(x, max_delta=0.05)
    return image, y

# train_ds = train_ds.map(augment)
'''




