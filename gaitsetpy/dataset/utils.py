'''
    This file contains the utility functions to download and extract the datasets.
    Supported datasets:
    - Daphnet
    
Maintainer: @aharshit123456
'''

## imports
import os
import requests
import zipfile
import tarfile
import json
import pandas as pd
import numpy as np
from glob import glob

#################################################################################
############################## DATASET DOWNLOAD #################################
#################################################################################

def download_dataset(dataset_name, data_dir):
    """Download the dataset."""
    if dataset_name == "daphnet":
        download_daphnet_data(data_dir)
    elif dataset_name == "mobifall":
        download_mobifall_data(data_dir)
    elif dataset_name == "arduous":
        download_arduous_data(data_dir)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    

def download_daphnet_data(data_dir):
    """Download the Daphnet dataset."""
    url = " https://archive.ics.uci.edu/static/public/245/daphnet+freezing+of+gait.zip"
    file_path = os.path.join(data_dir, "daphnet.zip")
    response = requests.get(url, stream=True)
    with open(file_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=128):
            file.write(chunk)

def download_mobifall_data(data_dir):
    """Download the MobiFall dataset."""
    pass

def download_arduous_data(data_dir):
    """Download the Arduous dataset."""
    pass


#################################################################################
############################## EXTRACT DOWNLOAD #################################
#################################################################################

def extract_dataset(dataset_name, data_dir):
    """Extract the dataset."""
    if dataset_name == "daphnet":
        extract_daphnet_data(data_dir)
    elif dataset_name == "mobifall":
        extract_mobifall_data(data_dir)
    elif dataset_name == "arduous":
        extract_arduous_data(data_dir)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    

def extract_daphnet_data(data_dir):
    """Extract the Daphnet dataset."""
    file_path = os.path.join(data_dir, "daphnet.zip")
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)

def extract_mobifall_data(data_dir):
    """Extract the MobiFall dataset."""
    pass

def extract_arduous_data(data_dir):
    """Extract the Arduous dataset."""
    pass


#################################################################################
############################ OTHER UTILS DOWNLOAD ###############################
#################################################################################


def sliding_window(data, window_size, step_size):
    num_windows = (len(data) - window_size) // step_size + 1
    windows = []
    for i in range(num_windows):
        start = i * step_size
        end = start + window_size
        windows.append(data[start:end])
    return windows