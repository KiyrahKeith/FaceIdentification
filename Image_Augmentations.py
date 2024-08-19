import os
import shutil

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import save_img
import numpy as np
import TensorBoard_Utils as TB_Utils
import Train_Models

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TB_Utils.reset_directory()  # Clear the previous TensorBoard log directory

dir = (Train_Models.dataset_dict["AFD Golden"])[0]  # The root of the directory to perform image augmentations
aug_file = "augmented"  # The name of the file in each subdirectory where the augmented images will be stored


# Deletes and recreates all files named with the aug_file name inside the root directory
def reset_augmented_folders():
    # Walk through the directory structure
    for dirpath, dirnames, filenames in os.walk(dir, topdown=False):
        if dirpath == dir: # Skip the root directory
            continue

        # Check if "augmented" file already exists in the current directory
        if "augmented" in dirnames:
            augmented_folder_path = os.path.join(dirpath, aug_file)
            shutil.rmtree(augmented_folder_path)

        # Create the "augmented" file
        augmented_folder_path = os.path.join(dirpath, aug_file)
        os.makedirs(augmented_folder_path)


# Creates a TensorBoard grid that shows multiple image augmentations at once.
def show_sample_augmentations():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dir,
        labels='inferred',
        label_mode="int"
    )
    train_ds = train_ds.map(TB_Utils.normalize_images)

    images, labels = next(iter(train_ds))
    for i in range(5):
        TB_Utils.sample_augmentations(images[i], str(i))


# Apply augmentations and save the augmented images for all files in the specified directory.
def augment():
    reset_augmented_folders()  # Ensure that augmented folders are empty and ready to be filled.

    dataset = tf.keras.utils.image_dataset_from_directory(
        dir,
        labels='inferred',
        image_size=(224, 224),
        label_mode="int"
    )
    class_names = dataset.class_names
    dataset = dataset.map(TB_Utils.normalize_images)

    i = 0
    for images, labels in dataset:
        for image, label in zip(images, labels):
            aug_path = os.path.join(dir, str(class_names[label.numpy()]), aug_file)
            image = tf.image.resize(image, [224, 224])

            # Horizontal flip
            flipped_image = tf.image.flip_left_right(image)
            save_img(os.path.join(aug_path, str(i)+'_flip.png'), flipped_image)

            # Rotate
            rotate_image = keras.layers.RandomRotation(0.10)(image)
            save_img(os.path.join(aug_path, str(i)+'_rotate.png'), rotate_image)

            # Hue
            hue_image = tf.image.random_hue(image, max_delta=0.3)
            save_img(os.path.join(aug_path, str(i)+'_hue.png'), hue_image)

            # Contrast
            contrast_image = tf.image.random_contrast(image, lower=2.5, upper=2.6)
            save_img(os.path.join(aug_path, str(i)+'_contrast.png'), contrast_image)

            # Saturation
            saturation_image = tf.image.random_saturation(image, lower=0.2, upper=0.4)
            save_img(os.path.join(aug_path, str(i)+'_saturation.png'), saturation_image)
            i += 1


if __name__ == "__main__":
    reset_augmented_folders()
    # show_sample_augmentations()
    # augment()
