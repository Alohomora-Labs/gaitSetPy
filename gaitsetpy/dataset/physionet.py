'''
PhysioNet VGRF Dataset Loader.
Maintainer: @aharshit123456

This file contains the PhysioNet VGRF dataset loader class that inherits from BaseDatasetLoader.
The PhysioNet dataset contains vertical ground reaction force (VGRF) data from subjects with 
Parkinson's disease and healthy controls.

Dataset source: https://physionet.org/content/gaitpdb/1.0.0/
'''

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from glob import glob
import requests
from tqdm import tqdm
import zipfile
from ..core.base_classes import BaseDatasetLoader
from .utils import sliding_window


class PhysioNetLoader(BaseDatasetLoader):
    """
    PhysioNet VGRF dataset loader class.
    
    This class handles loading and processing of the PhysioNet Gait in Parkinson's Disease dataset.
    The dataset contains vertical ground reaction force (VGRF) data from subjects with Parkinson's 
    disease and healthy controls.
    
    Features concurrent downloading for efficient data retrieval.
    """
    
    def __init__(self, max_workers: int = 8):
        """
        Initialize PhysioNet loader with concurrent download support.
        
        Args:
            max_workers: Maximum number of concurrent download threads (default: 8)
        """
        super().__init__(
            name="physionet",
            description="PhysioNet Gait in Parkinson's Disease Dataset - Contains VGRF data from subjects with Parkinson's disease and healthy controls",
            max_workers=max_workers
        )
        self.metadata = {
            'sensors': ['VGRF_L1', 'VGRF_L2', 'VGRF_L3', 'VGRF_L4', 'VGRF_L5', 'VGRF_L6', 'VGRF_L7', 'VGRF_L8',
                       'VGRF_R1', 'VGRF_R2', 'VGRF_R3', 'VGRF_R4', 'VGRF_R5', 'VGRF_R6', 'VGRF_R7', 'VGRF_R8'],
            'sampling_frequency': 100,  # 100 Hz sampling frequency
            'subjects': {
                'Co': 'Control subjects',
                'Pt': 'Parkinson\'s disease patients'
            },
            'window_size': 600,  # 6 seconds at 100 Hz
            'url': 'https://physionet.org/files/gaitpdb/1.0.0/'
        }
        self.labels = []
        self.subject_types = []
    
    def _download_physionet_data(self, data_dir: str) -> str:
        """
        Download PhysioNet dataset if not already present using concurrent downloads.
        
        This method uses multi-threaded downloading to significantly speed up the
        download process for the 100+ files in the PhysioNet dataset.
        
        Args:
            data_dir: Directory to store the dataset
            
        Returns:
            Path to the downloaded/existing dataset directory
        """
        dataset_path = os.path.join(data_dir, "physionet_gaitpdb")
        
        if os.path.exists(dataset_path) and len(os.listdir(dataset_path)) > 0:
            print(f"PhysioNet dataset already exists at: {dataset_path}")
            return dataset_path
        
        os.makedirs(dataset_path, exist_ok=True)
        
        # Download the dataset files
        base_url = "https://physionet.org/files/gaitpdb/1.0.0/"
        
        # Get list of files (basic file names based on the reference)
        file_patterns = [
            # Control subjects - Ga prefix
            *[f"GaCo{i:02d}_{j:02d}.txt" for i in range(1, 18) for j in range(1, 3)],
            "GaCo22_01.txt", "GaCo22_10.txt",
            
            # Parkinson's patients - Ga prefix
            *[f"GaPt{i:02d}_{j:02d}.txt" for i in range(3, 10) for j in range(1, 3)],
            *[f"GaPt{i:02d}_{j:02d}.txt" for i in range(12, 34) for j in range(1, 3)],
            *[f"GaPt{i:02d}_10.txt" for i in range(13, 34)],
            
            # Control subjects - Ju prefix
            *[f"JuCo{i:02d}_01.txt" for i in range(1, 27)],
            
            # Parkinson's patients - Ju prefix
            *[f"JuPt{i:02d}_{j:02d}.txt" for i in range(1, 30) for j in range(1, 8)],
            
            # Control subjects - Si prefix
            *[f"SiCo{i:02d}_01.txt" for i in range(1, 31)],
            
            # Parkinson's patients - Si prefix
            *[f"SiPt{i:02d}_01.txt" for i in range(2, 41)]
        ]
        
        # Prepare download tasks for concurrent execution
        download_tasks = [
            {
                'url': base_url + filename,
                'dest_path': os.path.join(dataset_path, filename)
            }
            for filename in file_patterns
        ]
        
        print(f"Downloading PhysioNet dataset to {dataset_path} using {self.max_workers} threads")
        
        # Use concurrent downloading from base class
        results = self.download_files_concurrent(
            download_tasks, 
            show_progress=True, 
            desc="Downloading PhysioNet files"
        )
        
        # Print summary
        print(f"\nDownload Summary:")
        print(f"  Total files: {results['total']}")
        print(f"  Successfully downloaded: {results['success']}")
        print(f"  Already existed (skipped): {results['skipped']}")
        print(f"  Failed: {results['failed']}")
        
        if results['failed'] > 0 and len(results['failed_downloads']) > 0:
            print(f"\nFailed downloads (showing first 10):")
            for failed in results['failed_downloads'][:10]:
                print(f"  - {os.path.basename(failed['dest_path'])}: {failed['error']}")
            if len(results['failed_downloads']) > 10:
                print(f"  ... and {len(results['failed_downloads']) - 10} more failures")
        
        return dataset_path
    
    def load_data(self, data_dir: str, **kwargs) -> Tuple[List[pd.DataFrame], List[str]]:
        """
        Load PhysioNet VGRF dataset from the specified directory.
        
        Args:
            data_dir: Directory to store/find the dataset
            **kwargs: Additional arguments (unused for PhysioNet)
            
        Returns:
            Tuple of (data_list, names_list)
        """
        # Download dataset if needed
        dataset_path = self._download_physionet_data(data_dir)
        
        physionet_data = []
        physionet_names = []
        self.labels = []
        self.subject_types = []
        
        # Load all available files
        for filepath in sorted(glob(os.path.join(dataset_path, "Ga*.txt"))):
            filename = os.path.basename(filepath)
            
            # Extract subject type from filename
            if 'Co' in filename:
                subject_type = 'Control'
                label = 'Co'
            elif 'Pt' in filename:
                subject_type = 'Patient'
                label = 'Pt'
            else:
                continue  # Skip files that don't match expected pattern
            
            try:
                # Read the file - PhysioNet files are tab-delimited with variable columns
                # Column 0: time, Columns 1-16: VGRF sensors, additional columns may exist
                df = pd.read_csv(filepath, delimiter='\t', header=None)
                
                # Handle variable number of columns
                n_cols = min(df.shape[1], 19)  # Limit to 19 columns max
                df = df.iloc[:, :n_cols]
                
                # Create column names
                col_names = ['time']
                for i in range(1, n_cols):
                    if i <= 8:
                        col_names.append(f'VGRF_L{i}')
                    elif i <= 16:
                        col_names.append(f'VGRF_R{i-8}')
                    else:
                        col_names.append(f'sensor_{i}')
                
                df.columns = col_names
                
                # Set time as index
                df = df.set_index('time')
                
                # Add subject metadata
                df['subject_type'] = subject_type
                df['label'] = label
                
                physionet_data.append(df)
                physionet_names.append(filename)
                self.labels.append(label)
                self.subject_types.append(subject_type)
                
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
        
        # Store loaded data
        self.data = physionet_data
        self.names = physionet_names
        
        print(f"Loaded {len(physionet_data)} PhysioNet files")
        print(f"Subject distribution: {dict(zip(*np.unique(self.subject_types, return_counts=True)))}")
        
        return physionet_data, physionet_names
    
    def create_sliding_windows(self, data: List[pd.DataFrame], names: List[str], 
                             window_size: int = 600, step_size: int = 100) -> List[Dict]:
        """
        Create sliding windows from the PhysioNet dataset.
        
        Args:
            data: List of DataFrames containing PhysioNet data
            names: List of names corresponding to the data
            window_size: Size of the sliding window (default: 600 for 6 seconds at 100Hz)
            step_size: Step size for the sliding window (default: 100)
            
        Returns:
            List of dictionaries containing sliding windows for each DataFrame
        """
        windows_data = []
        
        for idx, df in enumerate(data):
            # Remove metadata columns for windowing
            sensor_columns = [col for col in df.columns if col.startswith('VGRF_') or col.startswith('sensor_')]
            df_sensors = df[sensor_columns]
            
            if df_sensors.empty or len(df_sensors) < window_size:
                continue
                
            windows = []
            
            # Create windows for each sensor
            for col in sensor_columns:
                try:
                    window_data = sliding_window(df_sensors[col].values, window_size, step_size)
                    windows.append({"name": col, "data": window_data})
                except Exception as e:
                    print(f"Error creating windows for {col} in {names[idx]}: {e}")
                    continue
            
            if windows:
                windows_data.append({
                    "name": names[idx],
                    "windows": windows,
                    "metadata": {
                        "subject_type": df['subject_type'].iloc[0] if 'subject_type' in df.columns else 'Unknown',
                        "label": df['label'].iloc[0] if 'label' in df.columns else 'Unknown',
                        "window_size": window_size,
                        "step_size": step_size,
                        "num_windows": len(windows[0]["data"]) if windows else 0
                    }
                })
        
        return windows_data
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported file formats for PhysioNet dataset.
        
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
            'sampling_frequency': self.metadata['sampling_frequency'],
            'window_size': self.metadata['window_size']
        }
    
    def get_subject_info(self) -> Dict[str, str]:
        """
        Get information about subjects in the dataset.
        
        Returns:
            Dictionary containing subject information
        """
        return self.metadata['subjects']
    
    def get_labels(self) -> List[str]:
        """
        Get labels for loaded data.
        
        Returns:
            List of labels corresponding to loaded data
        """
        return self.labels
    
    def filter_by_subject_type(self, subject_type: str) -> Tuple[List[pd.DataFrame], List[str]]:
        """
        Filter loaded data by subject type.
        
        Args:
            subject_type: 'Control' or 'Patient'
            
        Returns:
            Tuple of (filtered_data, filtered_names)
        """
        if not self.data:
            raise ValueError("No data loaded. Call load_data() first.")
        
        filtered_data = []
        filtered_names = []
        
        for i, df in enumerate(self.data):
            if df['subject_type'].iloc[0] == subject_type:
                filtered_data.append(df)
                filtered_names.append(self.names[i])
        
        return filtered_data, filtered_names


# Legacy function for backward compatibility
def load_physionet_data(data_dir: str) -> Tuple[List[pd.DataFrame], List[str]]:
    """
    Legacy function to load PhysioNet data.
    
    Args:
        data_dir: Directory containing the dataset
        
    Returns:
        Tuple of (data_list, names_list)
    """
    loader = PhysioNetLoader()
    return loader.load_data(data_dir)


def create_physionet_windows(data: List[pd.DataFrame], names: List[str], 
                           window_size: int = 600, step_size: int = 100) -> List[Dict]:
    """
    Legacy function to create sliding windows from PhysioNet data.
    
    Args:
        data: List of DataFrames
        names: List of names
        window_size: Size of sliding window
        step_size: Step size for sliding window
        
    Returns:
        List of sliding window dictionaries
    """
    loader = PhysioNetLoader()
    return loader.create_sliding_windows(data, names, window_size, step_size) 