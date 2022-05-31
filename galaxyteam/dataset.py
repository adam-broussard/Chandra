"""
This file contains functions for downloading the Goodreads dataset for
analysis and cleaning it.
"""

import os
import kaggle
from skimage.io import imread, imsave
from skimage import img_as_ubyte
from skimage.util import crop
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
                                      path=savedir, unzip=True)

    print('done.')


def resize_image(read_fp, save_fp, target_shape=(550, 625, 3),
                 anti_alias=False):
    """
    Reads an image, converts it to uint8, resizes it, and saves it to file.

    Args:
        read_fp (string): The filepath to read the image from.
        save_fp (string): The filepath to save the image to.
        target_shape (iter): The dimensions of the final saved image.
        anti_alias (bool): Whether or not to implement gaussian smoothing when
            shrinking the images to prevent aliasing
    """

    # Make the save folder if it doesn't already exist.
    os.makedirs(os.path.dirname(save_fp), exist_ok=True)

    image = imread(read_fp)

    # Shrink the image such that one axis is the correct size and then crop the
    # long axis

    # Figuring out which axis is which
    if image.shape[1]/image.shape[0] > target_shape[1]/target_shape[0]:
        downscale_axis = 0
        crop_axis = 1
    else:
        downscale_axis = 1
        crop_axis = 0

    # Resizing
    resize_factor = image.shape[downscale_axis]/target_shape[downscale_axis]

    resized = resize(image, (int(image.shape[0]/resize_factor),
                             int(image.shape[1]/resize_factor),
                             3), anti_alias=anti_alias)

    # Cropping
    crop1 = (resized.shape[crop_axis] - target_shape[crop_axis])//2
    if (resized.shape[crop_axis] - target_shape[crop_axis]) % 2 == 0:
        crop2 = crop1
    else:
        crop2 = crop1 + 1

    crop_list = [None, None, [0, 0]]
    crop_list[crop_axis] = [crop1, crop2]
    crop_list[downscale_axis] = [0, 0]

    cropped = crop(resized, crop_list)

    # Save to file
    imsave(save_fp, img_as_ubyte(cropped))
