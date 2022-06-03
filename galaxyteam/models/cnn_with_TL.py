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
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from tensorflow import keras

from galaxyteam.models.cnn import create_dataset
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

    model.compile(loss='binary_crossentropy',
                optimizer = keras.optimizers.Adam(),
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

    model = build_TL(finetune)

    ES = EarlyStopping(monitor='f1_score',
                       patience=10,
                       restore_best_weights=True)

    history = model.fit(train_dataset, epochs=epochs, verbose=1,
                        validation_data=val_dataset, callbacks=[ES])

    return history, model
    