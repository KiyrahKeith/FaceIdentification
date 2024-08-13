import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import Binta_Models
import TensorBoard_Utils as utils
import Models as models

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

utils.reset_directory()  # Clear the previous TensorBoard log directory
if os.path.exists('results.csv'):  # Clear the previous results.csv file
    os.remove('results.csv')

# Stores the information for each dataset currently available
# key = dataset name, value = (directory, number of classes)
dataset_dict = {
    "CZoo": ('Datasets/CZoo/Chimpanzee/', 24),
    "CTai": ('Datasets/CTai/Chimpanzee/', 72),
    "AFD Golden": ('Datasets/AFD/Rhinopithecus roxellanae/', 243)
}

# Model parameters--------------------------------------------------------------------------------------------
dataset = "CZoo"
directory, num_classes = dataset_dict[dataset]
batch_size = 32
epochs = 60
validation_split = 0.1
learning_rate = 0.001
optimizer = keras.optimizers.legacy.Adam
loss = keras.losses.SparseCategoricalCrossentropy()
metrics = ['accuracy',
           keras.metrics.TopKCategoricalAccuracy(k=5, name="top5")]
# --------------------------------------------------------------------------------------------------------------


def train():
    # for transfer_model in models.models_dict.keys():
    transfer_model = "InceptionV3"
    # Preprocess the data based on what transfer model is being used
    train, validate = models.preprocess_data(transfer_model, train_ds, validation_ds, num_classes)

    # Create the model
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
    models.train_model(model,
                        train_ds=train,
                        validation_ds=validate,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=2,
                        record_name=record_name,
                        record_data=record_data)


# Trains the models and saves the results
def train_binta_models():
    train, validate = models.preprocess_data("VGG16", train_ds, validation_ds, num_classes)

    model = Binta_Models.vgg16_3(num_classes=num_classes)

    # Set up the training cycle information to be recorded in results.csv
    record_name = "vgg16_3"
    record_data = {'learning_rate': 1e-4, 'batch_size': 32}

    # Train the model
    history = models.train_model(model,
                                 train_ds=train,
                                 validation_ds=validate,
                                 batch_size=32,
                                 epochs=100,
                                 verbose=2,
                                 record_name=record_name,
                                 record_data=record_data)


if __name__ == "__main__":
    # Load and normalize the dataset
    train_ds, validation_ds = models.load_data(directory, batch_size, validation_split)

    train()
