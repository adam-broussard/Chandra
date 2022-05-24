"""
This file contains functions for downloading the Goodreads dataset for
analysis and cleaning it.
"""

import os
import kaggle


def download_dataset(savedir='data/'):
    """
    Creates a data directory if it doesn't exist and downloads the dataset
    to that directory.

    Args:
        savedir (str): The directory where the data will be saved
    """

    print(f'Downloading data csv to "{savedir}"...', end='')
    os.makedirs(savedir, exist_ok=True)

    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('paultimothymooney/chest-xray-pneumonia',
                                      path=savedir, unzip=True)

    print('done.')
