'''
Daphnet Dataset Loader and Utils.
Maintainer: @aharshit123456

This file contains the Daphnet dataset loader class that inherits from BaseDatasetLoader.
'''

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from glob import glob
from ..core.base_classes import BaseDatasetLoader
from .utils import download_dataset, extract_dataset, sliding_window


class DaphnetLoader(BaseDatasetLoader):
    """
    Daphnet dataset loader class.
    
    This class handles loading and processing of the Daphnet dataset for gait analysis.
    """
    
    def __init__(self, max_workers: int = 8):
        """
        Initialize Daphnet loader with concurrent download support.
        
        Args:
            max_workers: Maximum number of concurrent download threads (default: 8)
        """
        super().__init__(
            name="daphnet",
            description="Daphnet Freezing of Gait Dataset - Contains accelerometer data from subjects with Parkinson's disease",
            max_workers=max_workers
        )
        self.metadata = {
            'sensors': ['shank', 'thigh', 'trunk'],
            'components': ['h_fd', 'v', 'h_l'],  # horizontal forward, vertical, horizontal lateral
            'sampling_frequency': 64,
            'annotations': {
                0: 'not_valid',
                1: 'no_freeze',
                2: 'freeze'
            }
        }
    
    def load_data(self, data_dir: str, **kwargs) -> Tuple[List[pd.DataFrame], List[str]]:
        """
        Load Daphnet dataset from the specified directory.
        
        Args:
            data_dir: Directory to store/find the dataset
            **kwargs: Additional arguments (unused for Daphnet)
            
        Returns:
            Tuple of (data_list, names_list)
        """
        # Download and extract if needed
        download_dataset("daphnet", data_dir)
        extract_dataset("daphnet", data_dir)
        
        file_path = os.path.join(data_dir, "dataset_fog_release/dataset")
        daphnet_data = []
        daphnet_names = []
        
        # Load all subject files
        for file in sorted(glob(os.path.join(file_path, "S*.txt"))):
            # Extract filename from path
            filename = os.path.basename(file)
            daphnet_names.append(filename)
            
            # Load CSV with proper column names
            column_names = [
                "time", "shank_h_fd", "shank_v", "shank_h_l", 
                "thigh_h_fd", "thigh_v", "thigh_h_l", 
                "trunk_h_fd", "trunk_v", "trunk_h_l", "annotations"
            ]
            
            df = pd.read_csv(file, sep=" ", names=column_names)
            
            # Set time as index
            df = df.set_index("time")
            
            # Calculate magnitude for each sensor
            df["thigh"] = np.sqrt(df["thigh_h_l"]**2 + df["thigh_v"]**2 + df["thigh_h_fd"]**2)
            df["shank"] = np.sqrt(df["shank_h_l"]**2 + df["shank_v"]**2 + df["shank_h_fd"]**2)
            df["trunk"] = np.sqrt(df["trunk_h_l"]**2 + df["trunk_v"]**2 + df["trunk_h_fd"]**2)
            
            # Reorder columns for consistency
            df = df[["shank", "shank_h_fd", "shank_v", "shank_h_l", 
                    "thigh", "thigh_h_fd", "thigh_v", "thigh_h_l", 
                    "trunk", "trunk_h_fd", "trunk_v", "trunk_h_l", "annotations"]]
            
            daphnet_data.append(df)
        
        # Store loaded data
        self.data = daphnet_data
        self.names = daphnet_names
        
        return daphnet_data, daphnet_names
    
    def create_sliding_windows(self, data: List[pd.DataFrame], names: List[str], 
                             window_size: int = 192, step_size: int = 32) -> List[Dict]:
        """
        Create sliding windows from the Daphnet dataset.
        
        Args:
            data: List of DataFrames containing Daphnet data
            names: List of names corresponding to the data
            window_size: Size of the sliding window (default: 192)
            step_size: Step size for the sliding window (default: 32)
            
        Returns:
            List of dictionaries containing sliding windows for each DataFrame
        """
        windows_data = []
        
        for idx, df in enumerate(data):
            # Filter out invalid data (annotations == 0)
            df_filtered = df[df.annotations > 0]
            
            if df_filtered.empty:
                continue
                
            windows = []
            processed_columns = set()
            
            # Process each sensor column
            for col in df_filtered.columns:
                if col != "annotations" and col not in processed_columns:
                    window_data = sliding_window(df_filtered[col], window_size, step_size)
                    windows.append({"name": col, "data": window_data})
                    processed_columns.add(col)
            
            # Include annotations separately
            annotations_window = sliding_window(df_filtered["annotations"], window_size, step_size)
            windows.append({"name": "annotations", "data": annotations_window})
            
            windows_data.append({"name": names[idx], "windows": windows})
        
        return windows_data
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported file formats for Daphnet dataset.
        
        Returns:
            List of supported file extensions
        """
        return ['.txt']
    
    def get_sensor_info(self) -> Dict[str, List[str]]:
        """
        Get information about sensors in the dataset.
        
        Returns:
            Dictionary containing sensor information
        """
        return {
            'sensors': self.metadata['sensors'],
            'components': self.metadata['components'],
            'sampling_frequency': self.metadata['sampling_frequency']
        }
    
    def get_annotation_info(self) -> Dict[int, str]:
        """
        Get information about annotations in the dataset.
        
        Returns:
            Dictionary mapping annotation values to descriptions
        """
        return self.metadata['annotations']


# Legacy function wrappers for backward compatibility
def load_daphnet_data(data_dir: str):
    """
    Legacy function for loading Daphnet data.
    
    Args:
        data_dir: Directory to store the dataset
        
    Returns:
        Tuple of (data_list, names_list)
    """
    loader = DaphnetLoader()
    return loader.load_data(data_dir)


def create_sliding_windows(daphnet, daphnet_names, window_size=192, step_size=32):
    """
    Legacy function for creating sliding windows.
    
    Args:
        daphnet: List of dataframes containing Daphnet data
        daphnet_names: List of names of the Daphnet dataframes
        window_size: Size of the sliding window
        step_size: Step size for the sliding window
        
    Returns:
        List of dictionaries containing sliding windows for each DataFrame
    """
    loader = DaphnetLoader()
    return loader.create_sliding_windows(daphnet, daphnet_names, window_size, step_size)


def plot_dataset_sample():
    """Placeholder for dataset sample plotting."""
    pass


def plot_sliding_window():
    """Placeholder for sliding window plotting."""
    pass
