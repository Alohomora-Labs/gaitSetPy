"""
Base Dataset Class for GaitSetPy

This module provides the base dataset class that all dataset loaders should inherit from.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
from ..preprocessing.pipeline import PreprocessingPipeline
from ..features.gait_features import FeatureExtractor

class BaseDataset(ABC):
    """Base class for all dataset loaders."""
    
    def __init__(self, data_dir: str):
        """
        Initialize the dataset.
        
        Args:
            data_dir (str): Directory to store/load the dataset.
        """
        self.data_dir = data_dir
        self.data = None
        self.subject_ids = None
        self._preprocessor = PreprocessingPipeline()
        self._feature_extractor = FeatureExtractor()
    
    @abstractmethod
    def download(self) -> None:
        """Download the dataset from source."""
        pass
    
    @abstractmethod
    def load(self) -> Tuple[List[pd.DataFrame], List[str]]:
        """
        Load the dataset.
        
        Returns:
            Tuple[List[pd.DataFrame], List[str]]: List of dataframes and their names
        """
        pass
    
    def preprocess(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Preprocess the data using the preprocessing pipeline.
        
        Args:
            data (pd.DataFrame): Raw data to preprocess
            **kwargs: Additional preprocessing parameters
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        return self._preprocessor.process(data, **kwargs)
    
    def extract_features(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Extract features from preprocessed data.
        
        Args:
            data (pd.DataFrame): Preprocessed data
            **kwargs: Additional feature extraction parameters
            
        Returns:
            Dict[str, Any]: Extracted features
        """
        return self._feature_extractor.extract(data, **kwargs)
    
    def create_sliding_windows(self, data: pd.DataFrame, window_size: int, overlap: float = 0.5) -> List[pd.DataFrame]:
        """
        Create sliding windows from the data.
        
        Args:
            data (pd.DataFrame): Data to create windows from
            window_size (int): Size of each window
            overlap (float): Overlap between windows (0-1)
            
        Returns:
            List[pd.DataFrame]: List of windowed data
        """
        step = int(window_size * (1 - overlap))
        windows = []
        
        for i in range(0, len(data) - window_size + 1, step):
            window = data.iloc[i:i + window_size].copy()
            windows.append(window)
            
        return windows
    
    def get_subject_ids(self) -> List[str]:
        """Get list of subject IDs in the dataset."""
        return self.subject_ids if self.subject_ids is not None else []
    
    def __len__(self) -> int:
        """Get number of samples in the dataset."""
        return len(self.data) if self.data is not None else 0 