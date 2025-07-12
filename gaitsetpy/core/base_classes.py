"""
Base classes for GaitSetPy components.

This module defines abstract base classes that all components should inherit from.
Each base class defines the interface and common functionality for its respective component type.

Maintainer: @aharshit123456
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np


class BaseDatasetLoader(ABC):
    """
    Base class for all dataset loaders.
    
    All dataset loaders should inherit from this class and implement the required methods.
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize the dataset loader.
        
        Args:
            name: Name of the dataset
            description: Description of the dataset
        """
        self.name = name
        self.description = description
        self.data = None
        self.metadata = {}
    
    @abstractmethod
    def load_data(self, data_dir: str, **kwargs) -> Tuple[List[pd.DataFrame], List[str]]:
        """
        Load dataset from the specified directory.
        
        Args:
            data_dir: Directory containing the dataset
            **kwargs: Additional arguments specific to the dataset
            
        Returns:
            Tuple of (data_list, names_list)
        """
        pass
    
    @abstractmethod
    def create_sliding_windows(self, data: List[pd.DataFrame], names: List[str], 
                             window_size: int = 192, step_size: int = 32) -> List[Dict]:
        """
        Create sliding windows from the loaded data.
        
        Args:
            data: List of DataFrames
            names: List of names corresponding to the data
            window_size: Size of each sliding window
            step_size: Step size for sliding windows
            
        Returns:
            List of dictionaries containing sliding windows
        """
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported file formats.
        
        Returns:
            List of supported file extensions
        """
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the dataset.
        
        Returns:
            Dictionary containing dataset information
        """
        return {
            'name': self.name,
            'description': self.description,
            'metadata': self.metadata,
            'supported_formats': self.get_supported_formats()
        }


class BaseFeatureExtractor(ABC):
    """
    Base class for all feature extractors.
    
    All feature extractors should inherit from this class and implement the required methods.
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize the feature extractor.
        
        Args:
            name: Name of the feature extractor
            description: Description of the feature extractor
        """
        self.name = name
        self.description = description
        self.config = {}
    
    @abstractmethod
    def extract_features(self, windows: List[Dict], fs: int, **kwargs) -> List[Dict]:
        """
        Extract features from sliding windows.
        
        Args:
            windows: List of sliding window dictionaries
            fs: Sampling frequency
            **kwargs: Additional arguments for feature extraction
            
        Returns:
            List of feature dictionaries
        """
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Get names of features extracted by this extractor.
        
        Returns:
            List of feature names
        """
        pass
    
    def configure(self, config: Dict[str, Any]):
        """
        Configure the feature extractor.
        
        Args:
            config: Configuration dictionary
        """
        self.config.update(config)
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the feature extractor.
        
        Returns:
            Dictionary containing feature extractor information
        """
        return {
            'name': self.name,
            'description': self.description,
            'config': self.config,
            'feature_names': self.get_feature_names()
        }


class BasePreprocessor(ABC):
    """
    Base class for all preprocessors.
    
    All preprocessors should inherit from this class and implement the required methods.
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize the preprocessor.
        
        Args:
            name: Name of the preprocessor
            description: Description of the preprocessor
        """
        self.name = name
        self.description = description
        self.config = {}
        self.fitted = False
    
    @abstractmethod
    def fit(self, data: Union[pd.DataFrame, np.ndarray], **kwargs):
        """
        Fit the preprocessor to the data.
        
        Args:
            data: Input data to fit on
            **kwargs: Additional arguments for fitting
        """
        pass
    
    @abstractmethod
    def transform(self, data: Union[pd.DataFrame, np.ndarray], **kwargs) -> Union[pd.DataFrame, np.ndarray]:
        """
        Transform the data using the fitted preprocessor.
        
        Args:
            data: Input data to transform
            **kwargs: Additional arguments for transformation
            
        Returns:
            Transformed data
        """
        pass
    
    def fit_transform(self, data: Union[pd.DataFrame, np.ndarray], **kwargs) -> Union[pd.DataFrame, np.ndarray]:
        """
        Fit the preprocessor and transform the data.
        
        Args:
            data: Input data to fit and transform
            **kwargs: Additional arguments
            
        Returns:
            Transformed data
        """
        self.fit(data, **kwargs)
        return self.transform(data, **kwargs)
    
    def configure(self, config: Dict[str, Any]):
        """
        Configure the preprocessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config.update(config)
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the preprocessor.
        
        Returns:
            Dictionary containing preprocessor information
        """
        return {
            'name': self.name,
            'description': self.description,
            'config': self.config,
            'fitted': self.fitted
        }


class BaseEDAAnalyzer(ABC):
    """
    Base class for all EDA analyzers.
    
    All EDA analyzers should inherit from this class and implement the required methods.
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize the EDA analyzer.
        
        Args:
            name: Name of the EDA analyzer
            description: Description of the EDA analyzer
        """
        self.name = name
        self.description = description
        self.config = {}
    
    @abstractmethod
    def analyze(self, data: Union[pd.DataFrame, List[pd.DataFrame]], **kwargs) -> Dict[str, Any]:
        """
        Perform analysis on the data.
        
        Args:
            data: Input data to analyze
            **kwargs: Additional arguments for analysis
            
        Returns:
            Dictionary containing analysis results
        """
        pass
    
    @abstractmethod
    def visualize(self, data: Union[pd.DataFrame, List[pd.DataFrame]], **kwargs):
        """
        Create visualizations of the data.
        
        Args:
            data: Input data to visualize
            **kwargs: Additional arguments for visualization
        """
        pass
    
    def configure(self, config: Dict[str, Any]):
        """
        Configure the EDA analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config.update(config)
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the EDA analyzer.
        
        Returns:
            Dictionary containing EDA analyzer information
        """
        return {
            'name': self.name,
            'description': self.description,
            'config': self.config
        }


class BaseClassificationModel(ABC):
    """
    Base class for all classification models.
    
    All classification models should inherit from this class and implement the required methods.
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize the classification model.
        
        Args:
            name: Name of the classification model
            description: Description of the classification model
        """
        self.name = name
        self.description = description
        self.model = None
        self.config = {}
        self.trained = False
    
    @abstractmethod
    def train(self, features: List[Dict], **kwargs):
        """
        Train the classification model.
        
        Args:
            features: List of feature dictionaries
            **kwargs: Additional arguments for training
        """
        pass
    
    @abstractmethod
    def predict(self, features: List[Dict], **kwargs) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            features: List of feature dictionaries
            **kwargs: Additional arguments for prediction
            
        Returns:
            Array of predictions
        """
        pass
    
    @abstractmethod
    def evaluate(self, features: List[Dict], **kwargs) -> Dict[str, float]:
        """
        Evaluate the model performance.
        
        Args:
            features: List of feature dictionaries
            **kwargs: Additional arguments for evaluation
            
        Returns:
            Dictionary containing evaluation metrics
        """
        pass
    
    @abstractmethod
    def save_model(self, filepath: str):
        """
        Save the trained model to a file.
        
        Args:
            filepath: Path to save the model
        """
        pass
    
    @abstractmethod
    def load_model(self, filepath: str):
        """
        Load a trained model from a file.
        
        Args:
            filepath: Path to the saved model
        """
        pass
    
    def configure(self, config: Dict[str, Any]):
        """
        Configure the classification model.
        
        Args:
            config: Configuration dictionary
        """
        self.config.update(config)
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the classification model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'name': self.name,
            'description': self.description,
            'config': self.config,
            'trained': self.trained
        } 