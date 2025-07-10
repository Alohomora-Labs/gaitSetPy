"""
Daphnet Dataset Loader

This module provides a class-based implementation for loading and processing the Daphnet dataset.
"""

import os
import pandas as pd
import numpy as np
from glob import glob
from typing import List, Tuple, Dict, Any
from .base import BaseDataset
from .utils import download_dataset, extract_dataset

class DaphnetDataset(BaseDataset):
    """Daphnet dataset loader."""
    
    def __init__(self, data_dir: str):
        """
        Initialize the Daphnet dataset.
        
        Args:
            data_dir (str): Directory to store/load the dataset.
        """
        super().__init__(data_dir)
        self.dataset_name = "daphnet"
        self.file_path = os.path.join(data_dir, "dataset_fog_release/dataset")
        
    def download(self) -> None:
        """Download and extract the Daphnet dataset."""
        download_dataset(self.dataset_name, self.data_dir)
        extract_dataset(self.dataset_name, self.data_dir)
        
    def load(self) -> Tuple[List[pd.DataFrame], List[str]]:
        """
        Load the Daphnet dataset.
        
        Returns:
            Tuple[List[pd.DataFrame], List[str]]: List of dataframes and their names
        """
        daphnet = []
        daphnetNames = []
        
        for i in sorted(glob(os.path.join(self.file_path, "S*.txt"))):
            daphnetNames.append(os.path.basename(i))
            csv = pd.read_csv(
                i, 
                sep=" ", 
                names=[
                    "time", "shank_h_fd", "shank_v", "shank_h_l",
                    "thigh_h_fd", "thigh_v", "thigh_h_l",
                    "trunk_h_fd", "trunk_v", "trunk_h_l", "annotations"
                ]
            ).set_index("time")
            
            # Calculate magnitudes
            csv["thigh"] = np.sqrt(csv["thigh_h_l"]**2 + csv["thigh_v"]**2 + csv["thigh_h_fd"]**2)
            csv["shank"] = np.sqrt(csv["shank_h_l"]**2 + csv["shank_v"]**2 + csv["shank_h_fd"]**2)
            csv["trunk"] = np.sqrt(csv["trunk_h_l"]**2 + csv["trunk_v"]**2 + csv["trunk_h_fd"]**2)
            
            # Reorder columns
            csv = csv[[
                "shank", "shank_h_fd", "shank_v", "shank_h_l",
                "thigh", "thigh_h_fd", "thigh_v", "thigh_h_l",
                "trunk", "trunk_h_fd", "trunk_v", "trunk_h_l",
                "annotations"
            ]]
            
            daphnet.append(csv)
            
        self.data = daphnet
        self.subject_ids = daphnetNames
        return daphnet, daphnetNames
    
    def get_sensor_data(self, subject_idx: int) -> pd.DataFrame:
        """
        Get sensor data for a specific subject.
        
        Args:
            subject_idx (int): Index of the subject
            
        Returns:
            pd.DataFrame: Sensor data for the subject
        """
        if self.data is None:
            self.load()
        return self.data[subject_idx]
    
    def get_annotations(self, subject_idx: int) -> pd.Series:
        """
        Get annotations for a specific subject.
        
        Args:
            subject_idx (int): Index of the subject
            
        Returns:
            pd.Series: Annotations for the subject
        """
        if self.data is None:
            self.load()
        return self.data[subject_idx]["annotations"]
    
    def get_subject_name(self, subject_idx: int) -> str:
        """
        Get the name/ID of a specific subject.
        
        Args:
            subject_idx (int): Index of the subject
            
        Returns:
            str: Subject name/ID
        """
        if self.subject_ids is None:
            self.load()
        return self.subject_ids[subject_idx]


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
        windows = []
        processed_columns = set()
        
        for col in df.columns:
            if col != "annotations" and col not in processed_columns:
                window_data = sliding_window(df[col], window_size, step_size)
                windows.append({"name": col, "data": window_data})
                processed_columns.add(col)
        
        # Include the annotations column separately
        annotations_window = sliding_window(df["annotations"], window_size, step_size)
        windows.append({"name": "annotations", "data": annotations_window})
        
        daphnet_windows.append({"name": daphnetNames[idx], "windows": windows})

    return daphnet_windows


def plot_dataset_sample():
    pass

def plot_sliding_window():
    pass
