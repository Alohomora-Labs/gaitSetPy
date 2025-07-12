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
    """Download the Daphnet dataset.
    
    This function downloads the Daphnet Freezing of Gait dataset from the UCI Machine Learning Repository.
    It shows a progress bar during download and handles various potential errors.
    If the file already exists in the specified directory, it skips the download.
    
    Args:
        data_dir (str): Directory where the dataset will be downloaded
        
    Returns:
        str: Path to the downloaded file
        
    Raises:
        ConnectionError: If unable to connect to the download URL
        IOError: If unable to create or write to the download directory/file
        Exception: For other unexpected errors during download
    """
    import os
    import requests
    from tqdm import tqdm
    
    url = "https://archive.ics.uci.edu/static/public/245/daphnet+freezing+of+gait.zip"
    file_path = os.path.join(data_dir, "daphnet.zip")
    
    # Check if file already exists
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        print(f"Dataset already exists at: {file_path}")
        return file_path
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        print(f"Downloading Daphnet dataset to: {file_path}")
        
        # Send a HEAD request first to get the file size
        response = requests.head(url)
        total_size = int(response.headers.get('content-length', 0))
        
        # Start the download with progress bar
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Initialize progress bar
        progress_bar = tqdm(
            total=total_size,
            unit='iB',
            unit_scale=True,
            desc='Download Progress'
        )
        
        # Write the file with progress updates
        with open(file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    size = file.write(chunk)
                    progress_bar.update(size)
        
        progress_bar.close()
        
        # Verify download completed successfully
        if os.path.getsize(file_path) > 0:
            print(f"Download completed successfully! File saved to: {file_path}")
            return file_path
        else:
            raise IOError("Downloaded file is empty")
            
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to download URL: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)  # Clean up partial download
        raise ConnectionError(f"Failed to download dataset: {e}")
        
    except IOError as e:
        print(f"Error writing download file: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)  # Clean up partial download
        raise IOError(f"Failed to save dataset: {e}")
        
    except Exception as e:
        print(f"Unexpected error during download: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)  # Clean up partial download
        raise Exception(f"Download failed: {e}")

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