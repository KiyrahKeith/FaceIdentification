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
transfer_models = ["ResNet50", "InceptionV3", "VGG16"]
batch_size = 32
epochs = 2
validation_split = 0.1
learning_rate = 0.001
optimizer = keras.optimizers.legacy.Adam
loss = keras.losses.SparseCategoricalCrossentropy()
metrics = ['accuracy',
           keras.metrics.TopKCategoricalAccuracy(k=5, name="top5")]
# --------------------------------------------------------------------------------------------------------------

# Load and preprocess the dataset
train_ds, validation_ds = models.load_data(directory, batch_size, validation_split)
train_ds = train_ds.map(utils.normalize_images)
validation_ds = validation_ds.map(utils.normalize_images)

for transfer_model in transfer_models:
    # Create the model
    train, validate = models.preprocess_data(transfer_model, train_ds, validation_ds, num_classes)
    model = models.create_model(transfer_model,
                                learning_rate=learning_rate,
                                num_classes=num_classes,
                                optimizer=optimizer,
                                metric=metrics)


    print("Train ", transfer_model)

    # Set up the training cycle information to be recorded in results.csv
    record_name = transfer_model
    record_data = {'learning_rate': learning_rate, 'batch_size': batch_size}

    # Train the model
    history = models.train_model(model,
                                 train_ds=train,
                                 validation_ds=validate,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 verbose=2,
                                 record_name=record_name,
                                 record_data=record_data)
