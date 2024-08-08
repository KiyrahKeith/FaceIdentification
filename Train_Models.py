import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import TensorBoard_Utils as utils
import Models as models
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Stores the information for each dataset currently available
# key = dataset name, value = (directory, number of classes)
dataset_dict = {
    "CZoo": ('Datasets/CZoo/Chimpanzee/', 24),
    "CTai": ('Datasets/CTai/Chimpanzee/', 72)
}

# Model parameters--------------------------------------------------------------------------------------------
dataset = "CZoo"
directory, num_classes = dataset_dict[dataset]
transfer_model = "ResNet50"
batch_size = 32
epochs = 5
validation_split = 0.1
learning_rate = 0.001
optimizer = keras.optimizers.legacy.Adam
loss = keras.losses.SparseCategoricalCrossentropy()
# --------------------------------------------------------------------------------------------------------------

# Load and preprocess the dataset
train_ds, validation_ds = models.load_data(directory, batch_size, validation_split)
train_ds = train_ds.map(utils.normalize_images)
validation_ds = validation_ds.map(utils.normalize_images)
train_ds, validation_ds = models.preprocess_data(transfer_model, train_ds, validation_ds, num_classes)

# Create the model
model = models.create_model(transfer_model,
                            learning_rate=learning_rate,
                            num_classes=num_classes,
                            optimizer=optimizer)

# Train the model
history = models.train_model(model,
                             train_ds=train_ds,
                             validation_ds=validation_ds,
                             batch_size=batch_size,
                             epochs=epochs,
                             verbose=2)
