"""
This file contains functions for downloading the Goodreads dataset for
analysis and cleaning it.
"""

import os
import kaggle
from skimage.io import imread, imsave
from skimage import img_as_ubyte
from skimage.transform import resize


def download_dataset(savedir='./data/'):
    """
    Creates a data directory if it doesn't exist and downloads the dataset
    to that directory.

    Args:
        savedir (str): The directory where the data will be saved
    """

    print(f'Downloading images to "{os.path.abspath(savedir)}"...', end='')
    os.makedirs(savedir, exist_ok=True)

    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('paultimothymooney/chest-xray-pneumonia',
                                      path=savedir)

    print('done.')


def resize_image(read_fp, save_fp, resize_shape=(1100, 1250, 3)):
    """
    Reads an image, converts it to uint8, resizes it, and saves it to file.

    Args:
        read_fp (string): The filepath to read the image from.
        save_fp (string): The filepath to save the image to.
        resize_shape (iter): The dimensions of the final saved image.
    """

    # Make the save folder if it doesn't already exist.
    os.makedirs(os.path.dirname(save_fp), exist_ok=True)

    image = imread(read_fp)
    resized = resize(image, resize_shape)
    imsave(save_fp, img_as_ubyte(resized))
