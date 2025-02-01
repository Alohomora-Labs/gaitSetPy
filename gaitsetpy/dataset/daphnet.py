'''
Daphnet Dataset Loader and Utils.
Maintainer: @aharshit123456
'''

## imports
import os
import pandas as pd
import numpy as np
from utils import download_dataset, extract_dataset, sliding_window
from glob import glob


def load_daphnet_data(data_dir:str):
    """
    Load Daphnet dataset.
    Args:
        data_dir (str): Directory to store the dataset.
    Returns:
        daphnet (list): List of dataframes containing Daphnet data.
        daphnetNames (list): List of names of the Daphnet dataframes.
    
    """
    download_dataset("daphnet", data_dir)
    extract_dataset("daphnet", data_dir)

    file_path = os.path.join(data_dir,"dataset_fog_release/dataset")
    daphnet = []
    daphnetNames = []

    for i in sorted(glob(os.path.join(file_path, "S*.txt"))):
        daphnetNames.append(i.split("\\")[-1])
        csv = pd.read_csv(i, sep=" ", names=["time", "shank_h_fd", "shank_v", "shank_h_l", "thigh_h_fd", "thigh_v", "thigh_h_l", "trunk_h_fd", "trunk_v", "trunk_h_l", "annotations"])[["time", "shank_h_fd", "shank_v", "shank_h_l", "thigh_h_fd", "thigh_v", "thigh_h_l", "trunk_h_fd", "trunk_v", "trunk_h_l", "annotations"]].set_index("time")
        csv["thigh"] = np.sqrt(csv["thigh_h_l"]**2 + csv["thigh_v"]**2 + csv["thigh_h_fd"]**2)
        csv["shank"] = np.sqrt(csv["shank_h_l"]**2 + csv["shank_v"]**2 + csv["shank_h_fd"]**2)
        csv["trunk"] = np.sqrt(csv["trunk_h_l"]**2 + csv["trunk_v"]**2 + csv["trunk_h_fd"]**2)
        csv = csv[["shank", "shank_h_fd", "shank_v", "shank_h_l", "thigh", "thigh_h_fd", "thigh_v", "thigh_h_l", "trunk", "trunk_h_fd", "trunk_v", "trunk_h_l", "annotations"]]
        daphnet.append(csv)

    return daphnet, daphnetNames

def create_sliding_windows(daphnet, daphnetNames, window_size=192, step_size=32):
    """
    Create sliding windows from Daphnet dataset.
    Args:
        daphnet (list): List of dataframes containing Daphnet data.
        daphnetNames (list): List of names of the Daphnet dataframes.
        window_size (int): Size of the sliding window.
        step_size (int): Step size for the sliding window.
    Returns:
        daphnet_windows (list): List of dictionaries containing sliding windows for each DataFrame.
    """
    daphnet_windows = []

    for idx, df in enumerate(daphnet):
        ## skipping non includable data
        df = df[df.annotations > 0]
        df = df[df.annotations == 1 | df.annotations == 2]
        windows = []
        for col in df.columns:
            if col != "annotations":  # Exclude the annotations column from sliding windows
                window_data = sliding_window(df[col], window_size, step_size)
                windows.append({"name": col, "data": window_data})
        
        # Include the annotations column separately
        annotations_window = sliding_window(df["annotations"], window_size, step_size)
        windows.append({"name": "annotations", "data": annotations_window})
        
        daphnet_windows.append({"name": daphnetNames[idx], "windows": windows})

    return daphnet_windows


def plot_dataset_sample():
    pass

def plot_sliding_window():
    pass
