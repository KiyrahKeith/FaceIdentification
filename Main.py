import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import TensorBoard_Utils as utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

utils.reset_directory()  # Clear the previous TensorBoard log directory

# Initialize parameters
img_height = 128
img_width = 128
batch_size = 32
epochs = 10
num_classes = 24  # There are 24 individual chimps


model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)
])

# Create model layers
#model2 = keras.Sequential([
#    layers.Input((28, 28, 1)),
#    layers.Conv2D(16, 3, padding='same'),
#    layers.Conv2D(32, 3, padding='same'),
#    layers.MaxPooling2D(),
#    layers.Flatten(),
#    layers.Dense(10),
#])

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

# Load dataset from directory
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'Dataset/',
    labels='inferred',
    label_mode="int",
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset="training",
)

ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    'Dataset/',
    labels='inferred',
    label_mode="int",
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset="validation",
)

# Get the list of class names (if needed)
class_names = train_ds.class_names
# images, labels = next(iter(train_ds))
train_ds = train_ds.map(utils.normalize_images)
# train_ds = train_ds.map(augment)

# Create a sample grid of the training set images and load it into TensorBoard
# utils.image_grid("Training Set", train_ds, class_names)

images, labels = next(iter(train_ds))

for i in range(10):
    print("i: ", i)
    utils.sample_augmentations(images[i], str(i))



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


# Store the datasets in cache to reduce loading time.
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
ds_validation = ds_validation.cache().prefetch(buffer_size=AUTOTUNE)



# Set training settings
model.compile(
     optimizer=keras.optimizers.legacy.Adam(),
     loss=[
         keras.losses.SparseCategoricalCrossentropy(from_logits=True),
     ],
     metrics=["accuracy"],
 )
'''


'''
model.fit(train_ds, epochs=epochs, verbose=2)

history = model.fit(train_ds,
                    epochs=epochs,
                    validation_data=ds_validation)

# Plot training history
plt.figure(figsize=(12, 6))


# Plot training & validation loss values
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot training & validation accuracy values
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
'''


