'''
UrFall Dataset Loader and Utils.
Maintainer: @aharshit123456

This file contains the UrFall dataset loader class that inherits from BaseDatasetLoader.
UrFall is a fall detection dataset with multimodal data including depth, RGB, accelerometer,
and pre-extracted features from depth maps.

Reference:
- Website: https://fenix.ur.edu.pl/~mkepski/ds/uf.html
- Dataset: University of Rzeszow Fall Detection Dataset
'''

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from glob import glob
from ..core.base_classes import BaseDatasetLoader
from .utils import download_dataset, extract_dataset, sliding_window


class UrFallLoader(BaseDatasetLoader):
    """
    UrFall dataset loader class.
    
    This class handles loading and processing of the UrFall dataset for fall detection.
    Supports multiple data types: Depth, RGB, Accelerometer, Synchronization, Video,
    and pre-extracted features from depth maps.
    """
    
    def __init__(self):
        super().__init__(
            name="urfall",
            description="UrFall Dataset - University of Rzeszow Fall Detection Dataset with multimodal data"
        )
        self.metadata = {
            'data_types': ['depth', 'rgb', 'accelerometer', 'synchronization', 'video', 'features'],
            'camera': 'cam0',  # Front camera
            'sampling_frequency': 30,  # Depth/RGB camera fps
            'accelerometer_frequency': 100,  # Accelerometer sampling frequency (typical)
            'activities': {
                -1: 'Not lying (standing/walking)',
                0: 'Falling (transient)',
                1: 'Lying on ground'
            },
            'fall_sequences': list(range(1, 31)),  # fall-01 to fall-30
            'adl_sequences': list(range(1, 21)),  # adl-01 to adl-20
            'feature_columns': [
                'sequence_name',
                'frame_number',
                'label',
                'HeightWidthRatio',
                'MajorMinorRatio',
                'BoundingBoxOccupancy',
                'MaxStdXZ',
                'HHmaxRatio',
                'H',
                'D',
                'P40'
            ],
            'feature_descriptions': {
                'HeightWidthRatio': 'Bounding box height to width ratio',
                'MajorMinorRatio': 'Major to minor axis ratio from BLOB segmentation',
                'BoundingBoxOccupancy': 'Ratio of bounding box occupied by person pixels',
                'MaxStdXZ': 'Standard deviation of pixels from centroid (X and Z axis)',
                'HHmaxRatio': 'Human height in frame to standing height ratio',
                'H': 'Actual height in mm',
                'D': 'Distance of person center to floor in mm',
                'P40': 'Ratio of point clouds in 40cm cuboid to full height cuboid'
            }
        }
    
    def load_data(self, data_dir: str, 
                  data_types: Optional[List[str]] = None,
                  sequences: Optional[List[str]] = None,
                  use_falls: bool = True,
                  use_adls: bool = True,
                  **kwargs) -> Tuple[List[pd.DataFrame], List[str]]:
        """
        Load UrFall dataset from the specified directory.
        
        Args:
            data_dir: Directory containing the dataset
            data_types: List of data types to load. Options: 'depth', 'rgb', 'accelerometer',
                       'synchronization', 'video', 'features' (default: ['features'])
            sequences: List of specific sequences to load (e.g., ['fall-01', 'adl-01'])
                      If None, loads all based on use_falls and use_adls
            use_falls: Whether to load fall sequences (default: True)
            use_adls: Whether to load ADL (Activities of Daily Living) sequences (default: True)
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (data_list, names_list)
        """
        # Default to loading pre-extracted features if not specified
        if data_types is None:
            data_types = ['features']
        
        # Validate data types
        valid_types = set(self.metadata['data_types'])
        requested_types = set(data_types)
        invalid_types = requested_types - valid_types
        if invalid_types:
            raise ValueError(f"Invalid data types: {invalid_types}. Valid types: {valid_types}")
        
        # Create directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        data_list = []
        names_list = []
        
        # Load pre-extracted features (CSV files)
        if 'features' in data_types:
            features_data, features_names = self._load_features(data_dir, sequences, use_falls, use_adls)
            data_list.extend(features_data)
            names_list.extend(features_names)
        
        # Load raw accelerometer data
        if 'accelerometer' in data_types:
            accel_data, accel_names = self._load_accelerometer(data_dir, sequences, use_falls, use_adls)
            data_list.extend(accel_data)
            names_list.extend(accel_names)
        
        # Load synchronization data
        if 'synchronization' in data_types:
            sync_data, sync_names = self._load_synchronization(data_dir, sequences, use_falls, use_adls)
            data_list.extend(sync_data)
            names_list.extend(sync_names)
        
        # Note: Depth, RGB, and Video data are image/video files
        # These would require specialized loading and are not typically loaded into DataFrames
        if 'depth' in data_types or 'rgb' in data_types or 'video' in data_types:
            print("Note: Depth, RGB, and Video data types contain image/video files.")
            print("These are not loaded into DataFrames but their paths can be accessed.")
            print("Use the get_file_paths() method to retrieve paths to these files.")
        
        self.data = data_list
        return data_list, names_list
    
    def _load_features(self, data_dir: str, sequences: Optional[List[str]], 
                       use_falls: bool, use_adls: bool) -> Tuple[List[pd.DataFrame], List[str]]:
        """
        Load pre-extracted features from CSV files.
        
        Args:
            data_dir: Directory containing the dataset
            sequences: Specific sequences to load
            use_falls: Whether to include fall sequences
            use_adls: Whether to include ADL sequences
            
        Returns:
            Tuple of (data_list, names_list)
        """
        data_list = []
        names_list = []
        
        # Load falls features
        if use_falls:
            falls_csv = os.path.join(data_dir, "urfall-cam0-falls.csv")
            if os.path.exists(falls_csv):
                df = pd.read_csv(falls_csv, header=None, names=self.metadata['feature_columns'])
                
                # Filter by specific sequences if provided
                if sequences is not None:
                    fall_sequences = [s for s in sequences if s.startswith('fall-')]
                    if fall_sequences:
                        df = df[df['sequence_name'].isin(fall_sequences)]
                
                # Add metadata columns
                df['activity_type'] = 'fall'
                df['activity_id'] = 1  # Falls are labeled as 1
                
                data_list.append(df)
                names_list.append("urfall-cam0-falls")
            else:
                print(f"Warning: Falls features file not found at {falls_csv}")
        
        # Load ADLs features
        if use_adls:
            adls_csv = os.path.join(data_dir, "urfall-cam0-adls.csv")
            if os.path.exists(adls_csv):
                df = pd.read_csv(adls_csv, header=None, names=self.metadata['feature_columns'])
                
                # Filter by specific sequences if provided
                if sequences is not None:
                    adl_sequences = [s for s in sequences if s.startswith('adl-')]
                    if adl_sequences:
                        df = df[df['sequence_name'].isin(adl_sequences)]
                
                # Add metadata columns
                df['activity_type'] = 'adl'
                df['activity_id'] = 0  # ADLs are labeled as 0
                
                data_list.append(df)
                names_list.append("urfall-cam0-adls")
            else:
                print(f"Warning: ADLs features file not found at {adls_csv}")
        
        return data_list, names_list
    
    def _load_accelerometer(self, data_dir: str, sequences: Optional[List[str]],
                            use_falls: bool, use_adls: bool) -> Tuple[List[pd.DataFrame], List[str]]:
        """
        Load accelerometer CSV data files.
        
        Args:
            data_dir: Directory containing the dataset
            sequences: Specific sequences to load
            use_falls: Whether to include fall sequences
            use_adls: Whether to include ADL sequences
            
        Returns:
            Tuple of (data_list, names_list)
        """
        data_list = []
        names_list = []
        
        # Determine which sequences to load
        seq_list = []
        if sequences is not None:
            seq_list = sequences
        else:
            if use_falls:
                seq_list.extend([f"fall-{i:02d}" for i in range(1, 31)])
            if use_adls:
                seq_list.extend([f"adl-{i:02d}" for i in range(1, 21)])
        
        # Load accelerometer data for each sequence
        for seq in seq_list:
            accel_file = os.path.join(data_dir, f"{seq}-acc.csv")
            if os.path.exists(accel_file):
                try:
                    df = pd.read_csv(accel_file)
                    df['sequence_name'] = seq
                    df['activity_type'] = 'fall' if seq.startswith('fall-') else 'adl'
                    df['activity_id'] = 1 if seq.startswith('fall-') else 0
                    data_list.append(df)
                    names_list.append(f"{seq}-accelerometer")
                except Exception as e:
                    print(f"Warning: Could not load accelerometer data from {accel_file}: {e}")
        
        return data_list, names_list
    
    def _load_synchronization(self, data_dir: str, sequences: Optional[List[str]],
                              use_falls: bool, use_adls: bool) -> Tuple[List[pd.DataFrame], List[str]]:
        """
        Load synchronization CSV data files.
        
        Args:
            data_dir: Directory containing the dataset
            sequences: Specific sequences to load
            use_falls: Whether to include fall sequences
            use_adls: Whether to include ADL sequences
            
        Returns:
            Tuple of (data_list, names_list)
        """
        data_list = []
        names_list = []
        
        # Determine which sequences to load
        seq_list = []
        if sequences is not None:
            seq_list = sequences
        else:
            if use_falls:
                seq_list.extend([f"fall-{i:02d}" for i in range(1, 31)])
            if use_adls:
                seq_list.extend([f"adl-{i:02d}" for i in range(1, 21)])
        
        # Load synchronization data for each sequence
        for seq in seq_list:
            sync_file = os.path.join(data_dir, f"{seq}-data.csv")
            if os.path.exists(sync_file):
                try:
                    df = pd.read_csv(sync_file)
                    df['sequence_name'] = seq
                    df['activity_type'] = 'fall' if seq.startswith('fall-') else 'adl'
                    df['activity_id'] = 1 if seq.startswith('fall-') else 0
                    data_list.append(df)
                    names_list.append(f"{seq}-synchronization")
                except Exception as e:
                    print(f"Warning: Could not load synchronization data from {sync_file}: {e}")
        
        return data_list, names_list
    
    def get_file_paths(self, data_dir: str, data_type: str, 
                       sequences: Optional[List[str]] = None,
                       use_falls: bool = True, use_adls: bool = True) -> Dict[str, str]:
        """
        Get file paths for image/video data types (depth, RGB, video).
        
        Args:
            data_dir: Directory containing the dataset
            data_type: Type of data ('depth', 'rgb', 'video')
            sequences: Specific sequences to get paths for
            use_falls: Whether to include fall sequences
            use_adls: Whether to include ADL sequences
            
        Returns:
            Dictionary mapping sequence names to file paths
        """
        if data_type not in ['depth', 'rgb', 'video']:
            raise ValueError(f"data_type must be one of: 'depth', 'rgb', 'video'. Got: {data_type}")
        
        file_paths = {}
        
        # Determine which sequences to include
        seq_list = []
        if sequences is not None:
            seq_list = sequences
        else:
            if use_falls:
                seq_list.extend([f"fall-{i:02d}" for i in range(1, 31)])
            if use_adls:
                seq_list.extend([f"adl-{i:02d}" for i in range(1, 21)])
        
        # Map data type to file extension
        extension_map = {
            'depth': '-cam0-d.zip',
            'rgb': '-cam0-rgb.zip',
            'video': '-cam0.mp4'
        }
        
        ext = extension_map[data_type]
        
        for seq in seq_list:
            file_path = os.path.join(data_dir, f"{seq}{ext}")
            if os.path.exists(file_path):
                file_paths[seq] = file_path
        
        return file_paths
    
    def create_sliding_windows(self, data: List[pd.DataFrame], names: List[str],
                               window_size: int = 30, step_size: int = 15) -> List[Dict]:
        """
        Create sliding windows from the loaded data.
        
        Args:
            data: List of DataFrames containing the dataset
            names: List of names corresponding to each DataFrame
            window_size: Size of the sliding window (default: 30 frames for depth features)
            step_size: Step size for sliding window (default: 15 frames)
            
        Returns:
            List of dictionaries containing windowed data
        """
        windows_data = []
        
        for idx, df in enumerate(data):
            if df.empty:
                continue
            
            # Get numeric feature columns (exclude metadata columns)
            exclude_cols = ['sequence_name', 'frame_number', 'label', 'activity_type', 'activity_id']
            feature_cols = [col for col in df.columns 
                          if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
            
            if not feature_cols:
                continue
            
            windows = []
            
            # Create windows for each feature column
            for col in feature_cols:
                win = sliding_window(df[col].values, window_size, step_size)
                windows.append({"name": col, "data": win})
            
            # Create windows for labels if present
            if 'label' in df.columns:
                label_windows = sliding_window(df['label'].values, window_size, step_size)
                # Majority voting for each window
                labels = []
                for w in label_windows:
                    vals, counts = np.unique(w, return_counts=True)
                    labels.append(vals[np.argmax(counts)])
                windows.append({"name": "labels", "data": np.array(labels)})
            
            # Create activity_id windows
            if 'activity_id' in df.columns:
                activity_windows = sliding_window(df['activity_id'].values, window_size, step_size)
                windows.append({"name": "activity_id", "data": activity_windows})
            
            windows_data.append({"name": names[idx], "windows": windows})
        
        return windows_data
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported file formats for UrFall dataset.
        
        Returns:
            List of supported file extensions
        """
        return ['.csv', '.zip', '.mp4']
    
    def get_sensor_info(self) -> Dict[str, any]:
        """
        Get information about sensors in the dataset.
        
        Returns:
            Dictionary containing sensor information
        """
        return {
            'data_types': self.metadata['data_types'],
            'camera': self.metadata['camera'],
            'sampling_frequency': self.metadata['sampling_frequency'],
            'accelerometer_frequency': self.metadata['accelerometer_frequency']
        }
    
    def get_activity_info(self) -> Dict[int, str]:
        """
        Get information about activities in the dataset.
        
        Returns:
            Dictionary mapping activity IDs to labels
        """
        return self.metadata['activities']
    
    def get_feature_info(self) -> Dict[str, str]:
        """
        Get information about pre-extracted features.
        
        Returns:
            Dictionary mapping feature names to descriptions
        """
        return self.metadata['feature_descriptions']


# Legacy function wrappers for backward compatibility
def load_urfall_data(data_dir: str, data_types: Optional[List[str]] = None,
                     sequences: Optional[List[str]] = None,
                     use_falls: bool = True, use_adls: bool = True):
    """
    Load UrFall dataset using the legacy function interface.
    
    Args:
        data_dir: Directory containing the dataset
        data_types: List of data types to load
        sequences: List of specific sequences to load
        use_falls: Whether to load fall sequences
        use_adls: Whether to load ADL sequences
        
    Returns:
        Tuple of (data_list, names_list)
    """
    loader = UrFallLoader()
    return loader.load_data(data_dir, data_types=data_types, sequences=sequences,
                           use_falls=use_falls, use_adls=use_adls)


def create_urfall_windows(urfall_data, urfall_names, window_size=30, step_size=15):
    """
    Create sliding windows from UrFall data using the legacy function interface.
    
    Args:
        urfall_data: List of DataFrames
        urfall_names: List of names
        window_size: Size of sliding window
        step_size: Step size for sliding window
        
    Returns:
        List of dictionaries containing windowed data
    """
    loader = UrFallLoader()
    return loader.create_sliding_windows(urfall_data, urfall_names, window_size, step_size)
