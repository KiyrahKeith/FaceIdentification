import os
import tensorflow as tf
from keras.applications import InceptionV3
from tensorflow import keras
from tensorflow.keras import layers
from keras.optimizers import SGD, Adam, RMSprop
import Binta_Models
import TensorBoard_Utils as utils
import Models as models

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Stores the information for each dataset currently available
# key = dataset name, value = (directory, number of classes)
dataset_dict = {
    "CZoo": ('Datasets/CZoo/Chimpanzee/', 24),
    "CTai": ('Datasets/CTai/Chimpanzee/', 72),
    "AFD Golden": ('Datasets/AFD/Rhinopithecus roxellanae/', 243),
    "AFD": ('Datasets/AFD/', 18)
}

# Model parameters--------------------------------------------------------------------------------------------
dataset = "AFD"
directory, num_classes = dataset_dict[dataset]
batch_size = 32
epochs = 30
validation_split = 0.1
learning_rate = 0.001
optimizer = keras.optimizers.legacy.Adam
loss = keras.losses.SparseCategoricalCrossentropy()
metrics = ['accuracy']
# --------------------------------------------------------------------------------------------------------------


def train():
    transfer_model = "InceptionV3"
    # Preprocess the data based on what transfer model is being used
    train, validate = models.preprocess_data(transfer_model, train_ds, validation_ds)

    # Create the model
    model = models.create_model(transfer_model,
                                learning_rate=learning_rate,
                                num_classes=num_classes,
                                optimizer=optimizer(learning_rate=learning_rate),
                                metric=metrics)

    print("Train ", transfer_model)

    # Set up the training cycle information to be recorded in results.csv
    record_name = transfer_model
    record_data = {'learning_rate': learning_rate}

    # Train the model
    models.train_model(model,
                       train_ds=train,
                       validation_ds=validate,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=1,
                       record_name=record_name,
                       record_data=record_data)

    # predict = model.predict(validate, batch_size=batch_size, verbose=2)
    # print(f"Predict shape: {tf.shape(predict)}")


# Trains multiple models to systematically test various hyperparameters. Training results are stored in results.csv
def grid_search():
    # Search parameters
    transfer_models = ["InceptionV3"]
    learning_rates = [0.001]
    optimizers = [SGD, RMSprop]
    batch_sizes = [32]
    momentums = [0.0, 0.9]
    epochs = 30

    # Test each transfer model architecture. To test all available transfer architectures, use models.models_dict.keys()
    for transfer_model in transfer_models:
        # Preprocess the data based on what transfer model is being used
        train, validate = models.preprocess_data(transfer_model, train_ds, validation_ds)

        for momentum in momentums:
            train = train.unbatch().batch(batch_sizes[0])
            validate = validate.unbatch().batch(batch_sizes[0])

            for opt in optimizers:
                for lr in learning_rates:
                    # Create the model
                    model = models.create_model(transfer_model,
                                                learning_rate=lr,
                                                num_classes=num_classes,
                                                optimizer=opt(learning_rate=lr, momentum=momentum),
                                                metric=metrics)

                    # Set up the training cycle information to be recorded in results.csv
                    record_name = transfer_model + ", " + opt()._name + ", lr: " + str(lr) + ", momentum: " + str(momentum)
                    record_data = {'learning_rate': lr}

                    print("Train ", record_name)

                    # Train the model
                    models.train_model(model,
                                       train_ds=train,
                                       validation_ds=validate,
                                       batch_size=batch_size,
                                       epochs=epochs,
                                       verbose=1,
                                       record_name=record_name,
                                       record_data=record_data)


# Trains the models and saves the results
def train_binta_models():
    train, validate = models.preprocess_data("ResNet50", train_ds, validation_ds)

    model = Binta_Models.resnet_type1(num_classes=num_classes)

    # Set up the training cycle information to be recorded in results.csv
    record_name = "resnet_type1"
    record_data = {'learning_rate': 1e-4, 'batch_size': 32}

    # Train the model
    models.train_model(model,
                       train_ds=train,
                       validation_ds=validate,
                       batch_size=32,
                       epochs=100,
                       verbose=2,
                       record_name=record_name,
                       record_data=record_data)


if __name__ == "__main__":
    utils.reset_directory()  # Clear the previous TensorBoard log directory
    if os.path.exists('results.csv'):  # Clear the previous results.csv file
        os.remove('results.csv')

    # Load and normalize the dataset
    train_ds, validation_ds = models.load_data(directory, batch_size, validation_split)


    # grid_search()
    #train()
