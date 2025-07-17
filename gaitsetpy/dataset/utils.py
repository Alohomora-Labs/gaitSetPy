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
    elif dataset_name == "harup":
        download_harup_data(data_dir)
    elif dataset_name == "physionet":
        # PhysioNet dataset is handled by the PhysioNetLoader itself
        pass
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
    elif dataset_name == "harup":
        extract_harup_data(data_dir)
    elif dataset_name == "physionet":
        # PhysioNet dataset is handled by the PhysioNetLoader itself
        pass
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

def download_harup_data(data_dir):
    """
    Download the HAR-UP dataset.
    
    This function provides instructions for downloading the HAR-UP dataset and offers
    an option to download it directly from Google Drive as a ZIP file.
    
    Args:
        data_dir (str): Directory where the dataset will be downloaded
        
    Returns:
        str: Path to the extracted dataset directory or None if not performed
    """
    import os
    import requests
    from tqdm import tqdm
    import webbrowser
    import zipfile

    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Define file paths
    zip_filename = "HAR-UP_Dataset.zip"
    zip_path = os.path.join(data_dir, zip_filename)
    dataset_dir = os.path.join(data_dir, "DataSet")

    # Check if dataset directory already exists
    if os.path.exists(dataset_dir):
        print(f"HAR-UP dataset directory already exists at: {dataset_dir}")
        return dataset_dir

    # Direct download URL from Google Drive (update if needed)
    url = "https://drive.usercontent.google.com/download?id=1Y2MSUijPcB7--PcGoAKhGeqI8GxKK0Pm&export=download&authuser=0"
    print("\n" + "="*80)
    print("HAR-UP DATASET DOWNLOAD")
    print("="*80)
    print("The HAR-UP dataset can be downloaded automatically or manually.")
    print("\nOptions:")
    print("1. Automatic download (recommended)")
    print("2. Manual download")
    print("3. Skip download (if you already have the dataset elsewhere)")

    choice = input("\nEnter your choice (1-3): ")

    if choice == "1":
        try:
            print(f"\nDownloading HAR-UP dataset ZIP to: {zip_path}")
            print("This may take some time depending on your internet connection...")
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an exception for bad status codes
            total_size = int(response.headers.get('content-length', 0))
            progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc='Download Progress')
            with open(zip_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        size = file.write(chunk)
                        progress_bar.update(size)
            progress_bar.close()
            if os.path.getsize(zip_path) > 0:
                print(f"Download completed successfully! File saved to: {zip_path}")
                print("\nExtracting the downloaded ZIP file...")
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(data_dir)
                # Check for DataSet folder
                if not os.path.exists(dataset_dir):
                    # Sometimes the zip may contain a top-level folder, e.g., HAR-UP_Dataset/DataSet/...
                    for entry in os.listdir(data_dir):
                        entry_path = os.path.join(data_dir, entry)
                        if os.path.isdir(entry_path) and os.path.exists(os.path.join(entry_path, "DataSet")):
                            import shutil
                            shutil.move(os.path.join(entry_path, "DataSet"), dataset_dir)
                            break
                if os.path.exists(dataset_dir):
                    print(f"Extraction complete. DataSet directory at: {dataset_dir}")
                    return dataset_dir
                else:
                    print("Extraction failed: DataSet directory not found after extraction.")
                    return None
            else:
                raise IOError("Downloaded file is empty")
        except Exception as e:
            print(f"\nError during download: {e}")
            print("\nPlease try the manual download option instead.")
            if os.path.exists(zip_path):
                os.remove(zip_path)  # Clean up partial download
            return None

    elif choice == "2":
        print("\nOpening the HAR-UP dataset download page in your browser...")
        print("Please download the ZIP file and save it to the following location:")
        print(f"  {zip_path}")
        webbrowser.open("https://sites.google.com/up.edu.mx/har-up/download")
        print("\nAfter downloading, please ensure the ZIP file is named 'HAR-UP_Dataset.zip' and placed in your data directory.")
        print("Then, rerun this function or choose option 1 to extract.")
        return None

    elif choice == "3":
        print("\nSkipping download. Please ensure the dataset is available at:")
        print(f"  {os.path.join(data_dir, 'DataSet')}")
        return None

    else:
        print("\nInvalid choice. Please run again and select a valid option.")
        return None


def extract_harup_data(data_dir):
    """
    Extract the HAR-UP dataset zip file if not already extracted.
    """
    dataset_dir = os.path.join(data_dir, "DataSet")
    if os.path.exists(dataset_dir):
        print(f"HAR-UP dataset already extracted at: {dataset_dir}")
        return
    zip_path = os.path.join(data_dir, "HAR-UP_Dataset.zip")
    if not os.path.exists(zip_path):
        print(f"HAR-UP zip file not found at: {zip_path}")
        print("Please run download_harup_data first.")
        return
    import zipfile
    print(f"Extracting HAR-UP dataset zip to: {data_dir}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)
    print(f"Extraction complete.")