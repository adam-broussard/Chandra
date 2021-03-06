'''
Holds functions for building, training, saving, and reading convolutional
neural network models.
'''

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
# pylint: disable=[E0611,E0401]
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (Dense, Dropout, Flatten, Conv2D,
                                     MaxPooling2D)
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.metrics import Recall, Precision
import tensorflow as tf
# pylint: enable-[E0611,E0401]
from ..metrics import F1_Score


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

    lr_schedule = ExponentialDecay(
        0.0002,
        decay_steps=30,
        decay_rate=0.95)

    model.compile(loss=BinaryCrossentropy(),
                  optimizer=Adam(learning_rate=lr_schedule),
                  metrics=[Recall(),
                           Precision(),
                           F1_Score()])
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


def get_tf_train_val(train_info_file_path, batch_size=128, val_frac=0.2):
    """
    Returns tensorflow dataset objects for training and validation.

    Args:
        train_info_file_path (pathlib.Path or string): Location of the training
            metadata file
        batch_size(int): Size of mini-batches used for training
        val_frac (float): Fraction of the training set to use for validation

    Returns:
        train_dataset (tf.data.Dataset): A dataset containing the image and
            pneumonia data for training
        val_dataset (tf.data.Dataset): A dataset containing the image and
            pneumonia data for validation
    """

    full_train = pd.read_csv(train_info_file_path)

    train_data, val_data = train_test_split(full_train, test_size=val_frac,
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

    return train_dataset, val_dataset


def train_cnn(epochs=25, batch_size=32, val_frac=0.2,
              train_info_file_path=(Path('data')
                                    .joinpath('preprocessed',
                                              'train_metadata.csv'))):
    """
    Generates the necessary datasets and trains the CNN.

    Args:
        epochs (int): Number of training expochs
        batch_size(int): Size of mini-batches used for training
        val_frac (float): Fraction of the training set to use for validation
        train_info_file_path (pathlib.Path or string): Location of the training
            metadata file

    Returns:
        history (Tensorflow.History): The training history
        model (Tensorflow.Model): The trained model
    """

    train_dataset, val_dataset = get_tf_train_val(train_info_file_path,
                                                  batch_size=batch_size,
                                                  val_frac=val_frac)

    model = build_cnn()

    ES = EarlyStopping(monitor='val_f1_score',
                       patience=20,
                       restore_best_weights=True,
                       mode='max',
                       verbose=1)

    history = model.fit(train_dataset, epochs=epochs, verbose=1,
                        validation_data=val_dataset, callbacks=[ES])

    return history, model
