import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

img_size = 224  # The width and height dimensions of the image
batch_size = 32
epochs = 10
num_classes = 24  # There are 24 individual chimps in the C-Zoo dataset
validation_split = 0.2
optimizer = keras.optimizers.legacy.Adam(learning_rate=0.001)
loss = keras.losses.SparseCategoricalCrossentropy()
k_fold = 5

def badgernet():
    m = Sequential()
    m.add(layers.Input(shape=(img_size, img_size, 3)))
    m.add(layers.Conv2D(13, kernel_size=13, strides=4, activation='relu'))
    m.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2))
    m.add(layers.BatchNormalization())
    m.add(layers.Conv2D(50, kernel_size=5, padding="same", activation='relu'))
    m.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    m.add(layers.BatchNormalization())
    m.add(layers.Conv2D(80, kernel_size=3, padding="same", activation='relu'))
    m.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    m.add(layers.BatchNormalization())
    m.add(layers.Conv2D(100, kernel_size=3, padding="same", activation='relu'))
    m.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    m.add(layers.BatchNormalization())
    m.add(Flatten())
    m.add(Dense(1000, activation='relu'))
    m.add(layers.Dropout(0.5))
    m.add(keras.layers.Dense(num_classes, activation='softmax'))

    return m


train_ds, validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'Datasets/CZoo/Chimpanzee/',
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


model = badgernet()
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

model.summary()
history = model.fit(train_ds, batch_size=batch_size, epochs=epochs, verbose=2,
                    validation_data=validation_ds)