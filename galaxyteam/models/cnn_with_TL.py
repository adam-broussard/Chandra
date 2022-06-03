'''
Holds functions for building, training, saving, and reading
Transfer Learning based convolutional neural network models.
'''
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
# pylint: disable=[E0611,E0401]
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.layers import (Dense, Dropout, Input, GlobalAveragePooling2D)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from tensorflow import keras
# pylint: enable=[E0611,E0401]

from galaxyteam.models.cnn import create_dataset, get_tf_train_val
from galaxyteam.metrics import F1_Score



def build_TL(finetune = False):
    '''
    Builds the Transfer learning based CNN model
    Returns:
        model : TL model
    '''

    IMG_SIZE = 224
    ## We are using ResNet as out base model
    base_model = ResNet152V2(
                            weights='imagenet',
                            input_shape=(IMG_SIZE, IMG_SIZE, 3),
                            include_top=False)
    base_model.trainable = finetune
    #Input shape = [width, height, color channels]
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs)
    # Head
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    #Final Layer (Output)
    output = Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs=[inputs], outputs=output)
    keras.backend.clear_session()

    lr_schedule = ExponentialDecay(
        0.0002,
        decay_steps=30,
        decay_rate=0.95)

    model.compile(loss='binary_crossentropy',
                optimizer = keras.optimizers.Adam(lr_schedule),
                metrics=[Recall(),
                Precision(),
                F1_Score()])
    return model



def train_TL(finetune = False, epochs=10, batch_size=32, val_frac=0.2,
              train_info_file_path=(Path('data')
                                    .joinpath('preprocessed',
                                              'train_metadata.csv'))):
    """
    Generates the necessary datasets and trains the TL based CNN.
    Args:
        epochs (int): Number of training epochs
        batch_size(int): Size of mini-batches used for training
        val_frac (float): Fraction of the training set to use for validation
        train_info_file_path (pathlib.Path or string): Location of the training
            metadata file
    Returns:
        history (Tensorflow.History): The training history
        model (Tensorflow.Model): The trained model
    """

    train_dataset, val_dataset = get_tf_train_val(train_info_file_path,
                                                  val_frac=val_frac,
                                                  batch_size=batch_size)

    model = build_TL(finetune)

    ES = EarlyStopping(monitor='val_f1_score',
                       patience=20,
                       restore_best_weights=True,
                       mode='max',
                       verbose=1)

    history = model.fit(train_dataset, epochs=epochs, verbose=1,
                        validation_data=val_dataset, callbacks=[ES])

    return history, model
