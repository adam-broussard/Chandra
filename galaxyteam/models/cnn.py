'''
Holds functions for building, training, saving, and reading convolutional
neural network models.
'''

import json
import os
import pandas as pd
from tqdm import tqdm
from ..dataset import resize_image
from matplotlib import image
from PIL import UnidentifiedImageError
from sklearn.model_selection import train_test_split
# pylint: disable=[E0611,E0401]
from tensorflow.keras.models import Sequential, model_from_yaml
from tensorflow.keras.layers import (Dense, Dropout, Flatten, Conv2D, Lambda,
                                     MaxPooling2D)
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
# pylint: enable-[E0611,E0401]


def build_cnn():
    '''
    Builds the CNN model

    Returns:
        model (Sequential): CNN model
    '''

    model = Sequential([
                        Conv2D(64, 3,
                               activation='relu',
                               input_shape=(224, 224, 3)),
                        MaxPooling2D(4),
                        Conv2D(64, 3, activation='relu'),
                        MaxPooling2D(4),
                        Conv2D(32, 3, activation='relu'),
                        Flatten(),
                        Dense(64, activation='relu'),
                        Dropout(0.5),
                        Dense(16, activation='relu'),
                        Dropout(0.5),
                        Dense(1, activation='sigmoid')])

    model.compile(loss=BinaryCrossentropy(),
                  optimizer=Adam(),
                  metrics=[tf.keras.metrics.Recall(),
                           tf.keras.metrics.Precision()])
    return model


def _parse(file_name, is_pneumonia):
    '''
    Function that returns a tuple of normalized image array and pneumonia
    classification

    Args:
        file_name (string): Path to image
        is_pneumonia (bool): Boolean indicating whether the x-ray contains
            pneumonia

    returns:
        image_normalized (tf.Tensor): Normalized image tensor
        is_pneumonia (float): 1 or 0 indicating whether the x-ray contains
            pneumonia
    '''
    # Read an image from a file
    image_string = tf.io.read_file(file_name)
    # Decode it into a dense vector
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    # # Resize it to fixed shape
    # image_resized = resize_image(image_decoded, (224, 224, 3))
    return image_decoded, float(is_pneumonia)


def create_dataset(filenames, is_pneumonia, shuffle=False, batch_size=32):
    '''
    Create a tensorflow dataset object and return it.

    Args:
        filenames (iter): List of image paths
        is_pneumonia (iter): List of boolean values indicating if the x-ray
            contains pneumonia
        shuffle (bool): Whether or not to shuffle the dataset after generating
            it (note this is less effective than shuffling the filenames and
            ratings beforehand instead because the whole dataset cannot be
            stored in memory simultaneously.)
        batch_size (int): The number of images per batch

    Returns:
        dataset (tf.data.Dataset): A dataset containing the image and pneumonia
            data
    '''

    # Adapt preprocessing and prefetching dynamically to reduce GPU and CPU
    # idle time
    autotune = tf.data.experimental.AUTOTUNE

    # Create a first dataset of file paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((filenames, is_pneumonia))
    # Parse and preprocess observations in parallel
    dataset = dataset.map(_parse, num_parallel_calls=autotune)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=2048)

    # Batch the data for multiple steps
    dataset = dataset.batch(batch_size)
    # Fetch batches in the background while the model is training.
    dataset = dataset.prefetch(buffer_size=autotune)

    return dataset


def train_cnn(epochs=25, batch_size=32, val_frac=0.2,
              train_info_file_path='./data/preprocessed/train_metadata.csv'):

    full_train = pd.read_csv(train_info_file_path)

    train_data, val_data = train_test_split(full_train, test_size=0.2,
                                            shuffle=True,
                                            stratify=full_train.is_pneumonia,
                                            random_state=9473)

    # Create Tensorflow datasets
    train_dataset = create_dataset(train_data.resized_file_path,
                                   train_data.is_pneumonia,
                                   batch_size=batch_size)
    val_dataset = create_dataset(val_data.resized_file_path,
                                 val_data.is_pneumonia,
                                 batch_size=batch_size)

    model = build_cnn()

    history = model.fit(train_dataset, epochs=epochs, verbose=1,
                        validation_data=val_dataset)

    return history, model
