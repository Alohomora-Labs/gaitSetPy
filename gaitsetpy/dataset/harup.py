'''
HAR-UP Dataset Loader and Utils.
Maintainer: @aharshit123456

This file contains the HAR-UP dataset loader class that inherits from BaseDatasetLoader.
HAR-UP is a multimodal dataset for human activity recognition and fall detection.

Reference:
- Website: https://sites.google.com/up.edu.mx/har-up/
- GitHub: https://github.com/jpnm561/HAR-UP
'''

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from glob import glob
import datetime
from tqdm import tqdm
from ..core.base_classes import BaseDatasetLoader
from .utils import download_dataset, extract_dataset, sliding_window
from ..features.harup_features import HARUPFeatureExtractor


class HARUPLoader(BaseDatasetLoader):
    """
    HAR-UP dataset loader class.
    
    This class handles loading and processing of the HAR-UP dataset for human activity recognition
    and fall detection analysis.
    """
    
    def __init__(self, max_workers: int = 8):
        """
        Initialize HAR-UP loader with concurrent download support.
        
        Args:
            max_workers: Maximum number of concurrent download threads (default: 8)
        """
        super().__init__(
            name="harup",
            description="HAR-UP Dataset - Multimodal System for Fall Detection and Human Activity Recognition",
            max_workers=max_workers
        )
        self.metadata = {
            'sensors': [
                'AnkleAccelerometer', 'AnkleAngularVelocity', 'AnkleLuminosity',
                'RightPocketAccelerometer', 'RightPocketAngularVelocity', 'RightPocketLuminosity',
                'BeltAccelerometer', 'BeltAngularVelocity', 'BeltLuminosity',
                'NeckAccelerometer', 'NeckAngularVelocity', 'NeckLuminosity',
                'WristAccelerometer', 'WristAngularVelocity', 'WristLuminosity',
                'BrainSensor', 'Infrared'
            ],
            'components': {
                'Accelerometer': ['x', 'y', 'z'],
                'AngularVelocity': ['x', 'y', 'z'],
                'Luminosity': ['illuminance'],
                'BrainSensor': ['value'],
                'Infrared': ['value']
            },
            'sampling_frequency': 100,  # Hz
            'activities': {
                1: 'Walking',
                2: 'Walking upstairs',
                3: 'Walking downstairs',
                4: 'Sitting',
                5: 'Standing',
                6: 'Lying',
                7: 'Falling forward using hands',
                8: 'Falling forward using knees',
                9: 'Falling backwards',
                10: 'Falling sideward',
                11: 'Falling sitting in empty chair'
            }
        }
        
        # Features used in HAR-UP
        self.features = [
            'Mean', 'StandardDeviation', 'RootMeanSquare', 'MaximalAmplitude',
            'MinimalAmplitude', 'Median', 'Number of zero-crossing', 'Skewness',
            'Kurtosis', 'First Quartile', 'Third Quartile', 'Autocorrelation',
            'Energy'
        ]
    
    def download_harup_data(self, data_dir: str) -> Optional[str]:
        """
        Download HAR-UP dataset if not already present.
        
        Args:
            data_dir: Directory to store the dataset
            
        Returns:
            Path to the extracted dataset or None if not found
        """
        # Use the utility function to download and extract the dataset
        download_dataset("harup", data_dir)
        extract_dataset("harup", data_dir)
        
        # Check if dataset exists after download attempt
        dataset_path = os.path.join(data_dir, "DataSet")
        if not os.path.exists(dataset_path):
            print("HAR-UP dataset not found after download attempt.")
            print("Please ensure the dataset is organized in the following structure:")
            print("DataSet/Subject{i}/Activity{j}/Trial{k}/Subject{i}Activity{j}Trial{k}.csv")
            return None
        
        return dataset_path
    
    def load_data(self, data_dir: str, subjects: Optional[List[int]] = None, 
                activities: Optional[List[int]] = None, trials: Optional[List[int]] = None,
                **kwargs) -> Tuple[List[pd.DataFrame], List[str]]:
        """
        Load HAR-UP dataset from the specified directory.
        Args:
            data_dir: Directory containing the dataset
            subjects: List of subject IDs to load (default: all subjects)
            activities: List of activity IDs to load (default: all activities)
            trials: List of trial IDs to load (default: all trials)
            **kwargs: Additional arguments
        Returns:
            Tuple of (data_list, names_list)
        """
        import re
        import os
        # Set default values if not provided (HAR-UP: 4 subjects, 11 activities, 3 trials)
        if subjects is None:
            subjects = list(range(1, 5))  # 4 subjects
        if activities is None:
            activities = list(range(1, 12))  # 11 activities
        if trials is None:
            trials = list(range(1, 4))  # 3 trials

        # Column names as per official HAR-UP documentation
        columns = [
            "Timestamp",
            "EEG_NeuroSky",
            "Belt_Acc_X", "Belt_Acc_Y", "Belt_Acc_Z",
            "Belt_Gyro_X", "Belt_Gyro_Y", "Belt_Gyro_Z",
            "Belt_Luminosity",
            "Neck_Acc_X", "Neck_Acc_Y", "Neck_Acc_Z",
            "Neck_Gyro_X", "Neck_Gyro_Y", "Neck_Gyro_Z",
            "Neck_Luminosity",
            "Pocket_Acc_X", "Pocket_Acc_Y", "Pocket_Acc_Z",
            "Pocket_Gyro_X", "Pocket_Gyro_Y", "Pocket_Gyro_Z",
            "Pocket_Luminosity",
            "Wrist_Acc_X", "Wrist_Acc_Y", "Wrist_Acc_Z",
            "Wrist_Gyro_X", "Wrist_Gyro_Y", "Wrist_Gyro_Z",
            "Wrist_Luminosity",
            "Infrared_1", "Infrared_2", "Infrared_3", "Infrared_4"
        ]

        # If data_dir does not exist, trigger interactive download
        if not os.path.exists(data_dir):
            print(f"Directory {data_dir} does not exist. Attempting to download HAR-UP dataset...")
            self.download_harup_data(data_dir)
        # If still doesn't exist, error out
        if not os.path.exists(data_dir):
            print(f"Failed to create or download dataset directory: {data_dir}")
            return [], []

        # Find the UP_Fall_Detection_Dataset directory
        dataset_path = None
        for entry in os.listdir(data_dir):
            entry_path = os.path.join(data_dir, entry)
            if os.path.isdir(entry_path) and entry.startswith("UP_Fall_Detection_Dataset"):
                dataset_path = entry_path
                break
        if dataset_path is None:
            print("UP_Fall_Detection_Dataset directory not found in", data_dir)
            print("No data loaded. Please make sure you've downloaded the HAR-UP dataset.")
            print("Visit https://sites.google.com/up.edu.mx/har-up/ to download the dataset.")
            return [], []

        harup_data = []
        harup_names = []

        # Iterate over subjects
        for subject_id in subjects:
            subject_folder = f"Subject_{subject_id:02d}"
            subject_path = os.path.join(dataset_path, subject_folder)
            if not os.path.isdir(subject_path):
                continue
            
            # Initialize empty DataFrame for this subject
            subject_df = pd.DataFrame()
            
            # Iterate over activities in order
            for activity_id in sorted(activities):
                activity_folder = f"A{activity_id:02d}"
                activity_path = os.path.join(subject_path, activity_folder)
                if not os.path.isdir(activity_path):
                    continue
                
                # Iterate over trials in order
                for trial_id in sorted(trials):
                    file_name = f"S{subject_id:02d}_A{activity_id:02d}_T{trial_id:02d}.csv"
                    file_path = os.path.join(activity_path, file_name)
                    name = f"{subject_folder}_{activity_folder}_T{trial_id:02d}"
                    
                    try:
                        df = pd.read_csv(file_path, header=0)
                        print(f"[HARUP] Loaded columns for {file_name}: {list(df.columns)}")
                        df['subject_id'] = subject_id
                        df['activity_id'] = activity_id 
                        df['trial_id'] = trial_id
                        df['activity_label'] = self.metadata['activities'].get(activity_id, f"A{activity_id:02d}")
                        
                        # Concatenate to subject's DataFrame
                        subject_df = pd.concat([subject_df, df], ignore_index=True)
                        harup_names.append(name)
                        
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
            
            # Add complete subject DataFrame to data list
            if not subject_df.empty:
                harup_data.append(subject_df)
                
        self.data = harup_data
        self.names = harup_names

        return harup_data, harup_names
    
    def create_sliding_windows(self, data: List[pd.DataFrame], names: List[str], 
                             window_size: int = 100, step_size: int = 50) -> List[Dict]:
        """
        Create sliding windows from the HAR-UP dataset.
        
        Args:
            data: List of DataFrames containing HAR-UP data
            names: List of names corresponding to the data
            window_size: Size of the sliding window (default: 100 = 1 second at 100Hz)
            step_size: Step size for the sliding window (default: 50 = 0.5 seconds at 100Hz)
            
        Returns:
            List of dictionaries containing sliding windows for each DataFrame
        """
        windows_data = []
        
        for idx, df in enumerate(data):
            if df.empty:
                continue
                
            windows = []
            processed_columns = set()
            
            # Only use numeric columns (skip TIME and any non-numeric)
            sensor_columns = [col for col in df.columns if col not in 
                             ['subject_id', 'activity_id', 'trial_id', 'activity_label', 'TIME']
                             and pd.api.types.is_numeric_dtype(df[col])]
            

            # Process each sensor column
            for col in sensor_columns:
                if col not in processed_columns:
                    
                    window_data = sliding_window(df[col], window_size, step_size)
                    windows.append({"name": col, "data": window_data})
                    processed_columns.add(col)
            
            # Include activity ID for each window
            activity_windows = sliding_window(df["activity_id"], window_size, step_size)
            windows.append({"name": "activity_id", "data": activity_windows})
            
            # For each window, take the most common activity ID as the label
            labels = []
            for window in activity_windows:
                # Get most common activity in this window
                unique_vals, counts = np.unique(window, return_counts=True)
                most_common_idx = np.argmax(counts)
                labels.append(unique_vals[most_common_idx])
            
            windows.append({"name": "labels", "data": np.array(labels)})
            
            windows_data.append({"name": names[idx], "windows": windows})
        
        return windows_data
    
    def extract_features(self, windows_data: List[Dict], time_domain_features: bool = True,
                       freq_domain_features: bool = True) -> List[Dict]:
        """
        Extract features from sliding windows using HAR-UP feature extraction methods.
        Args:
            windows_data: List of dictionaries containing sliding windows
            time_domain_features: Whether to extract time domain features
            freq_domain_features: Whether to extract frequency domain features
        Returns:
            List of dictionaries containing extracted features
        """
        # Mapping from original sensor names to actual CSV column names
        sensor_map = {
            'BeltAccelerometer: x-axis (g)': 'BELT_ACC_X',
            'BeltAccelerometer: y-axis (g)': 'BELT_ACC_Y',
            'BeltAccelerometer: z-axis (g)': 'BELT_ACC_Z',
            'BeltAngularVelocity: x-axis (deg/s)': 'BELT_ANG_X',
            'BeltAngularVelocity: y-axis (deg/s)': 'BELT_ANG_Y',
            'BeltAngularVelocity: z-axis (deg/s)': 'BELT_ANG_Z',
            'BeltLuminosity: illuminance (lx)': 'BELT_LUMINOSITY',
            'NeckAccelerometer: x-axis (g)': 'NECK_ACC_X',
            'NeckAccelerometer: y-axis (g)': 'NECK_ACC_Y',
            'NeckAccelerometer: z-axis (g)': 'NECK_ACC_Z',
            'NeckAngularVelocity: x-axis (deg/s)': 'NECK_ANG_X',
            'NeckAngularVelocity: y-axis (deg/s)': 'NECK_ANG_Y',
            'NeckAngularVelocity: z-axis (deg/s)': 'NECK_ANG_Z',
            'NeckLuminosity: illuminance (lx)': 'NECK_LUMINOSITY',
            'PocketAccelerometer: x-axis (g)': 'PCKT_ACC_X',
            'PocketAccelerometer: y-axis (g)': 'PCKT_ACC_Y',
            'PocketAccelerometer: z-axis (g)': 'PCKT_ACC_Z',
            'PocketAngularVelocity: x-axis (deg/s)': 'PCKT_ANG_X',
            'PocketAngularVelocity: y-axis (deg/s)': 'PCKT_ANG_Y',
            'PocketAngularVelocity: z-axis (deg/s)': 'PCKT_ANG_Z',
            'PocketLuminosity: illuminance (lx)': 'PCKT_LUMINOSITY',
            'WristAccelerometer: x-axis (g)': 'WRST_ACC_X',
            'WristAccelerometer: y-axis (g)': 'WRST_ACC_Y',
            'WristAccelerometer: z-axis (g)': 'WRST_ACC_Z',
            'WristAngularVelocity: x-axis (deg/s)': 'WRST_ANG_X',
            'WristAngularVelocity: y-axis (deg/s)': 'WRST_ANG_Y',
            'WristAngularVelocity: z-axis (deg/s)': 'WRST_ANG_Z',
            'WristLuminosity: illuminance (lx)': 'WRST_LUMINOSITY',
            'BrainSensor': 'HELMET_RAW',
            'Infrared1': 'IR_1',
            'Infrared2': 'IR_2',
            'Infrared3': 'IR_3',
            'Infrared4': 'IR_4',
        }
        extractor = HARUPFeatureExtractor(verbose=True)
        extractor.config['time_domain'] = time_domain_features
        extractor.config['frequency_domain'] = freq_domain_features
        all_features = []
        for window_dict in windows_data:
            name = window_dict["name"]
            windows = window_dict["windows"]
            labels = None
            for window in windows:
                if window["name"] == "labels":
                    labels = window["data"]
                    break
            if labels is None:
                print(f"No labels found for {name}, skipping feature extraction")
                continue
            filtered_windows = []
            missing = []
            for orig_sensor, csv_col in sensor_map.items():
                found = False
                for window in windows:
                    if window["name"] == csv_col:
                        filtered_windows.append(window)
                        found = True
                        break
                if not found:
                    missing.append((orig_sensor, csv_col))
            if missing:
                print(f"[HARUP] Missing columns for {name}: {[m[1] for m in missing]}")
            for window in windows:
                if window["name"] == "activity_id" or window["name"] == "labels":
                    filtered_windows.append(window)
            features = extractor.extract_features(filtered_windows, fs=self.metadata['sampling_frequency'])
            for i, feature in enumerate(features):
                window_idx = i // (len(filtered_windows) - 2)  # Subtract 2 for labels and activity_id
                if window_idx < len(labels):
                    feature["label"] = labels[window_idx]
            all_features.append({"name": name, "features": features})
        return all_features
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported file formats for HAR-UP dataset.
        
        Returns:
            List of supported file extensions
        """
        return ['.csv']
    
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
    
    def get_activity_info(self) -> Dict[int, str]:
        """
        Get information about activities in the dataset.
        
        Returns:
            Dictionary mapping activity IDs to descriptions
        """
        return self.metadata['activities']


# Legacy function wrappers for backward compatibility
def load_harup_data(data_dir: str, subjects=None, activities=None, trials=None):
    """
    Legacy function for loading HAR-UP data.
    
    Args:
        data_dir: Directory containing the dataset
        subjects: List of subject IDs to load (default: all subjects)
        activities: List of activity IDs to load (default: all activities)
        trials: List of trial IDs to load (default: all trials)
        
    Returns:
        Tuple of (data_list, names_list)
    """
    loader = HARUPLoader()
    return loader.load_data(data_dir, subjects, activities, trials)


def create_harup_windows(harup_data, harup_names, window_size=100, step_size=50):
    """
    Legacy function for creating sliding windows from HAR-UP data.
    
    Args:
        harup_data: List of dataframes containing HAR-UP data
        harup_names: List of names of the HAR-UP dataframes
        window_size: Size of the sliding window
        step_size: Step size for the sliding window
        
    Returns:
        List of dictionaries containing sliding windows for each DataFrame
    """
    loader = HARUPLoader()
    return loader.create_sliding_windows(harup_data, harup_names, window_size, step_size)


def extract_harup_features(windows_data, time_domain=True, freq_domain=True):
    """
    Legacy function for extracting features from HAR-UP windows.
    
    Args:
        windows_data: List of dictionaries containing sliding windows
        time_domain: Whether to extract time domain features
        freq_domain: Whether to extract frequency domain features
        
    Returns:
        List of dictionaries containing extracted features
    """
    loader = HARUPLoader()
    return loader.extract_features(windows_data, time_domain, freq_domain)