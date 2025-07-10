"""
Base Model Class for GaitSetPy

This module provides the base model class that all classification models should inherit from.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from ..utils.preprocess import preprocess_features

class BaseModel(ABC):
    """Base class for all classification models."""
    
    def __init__(self):
        """Initialize the base model."""
        self.model = None
        self.is_trained = False
        
    @abstractmethod
    def build(self, **kwargs) -> None:
        """
        Build the model architecture.
        
        Args:
            **kwargs: Model-specific parameters
        """
        pass
    
    def preprocess_features(self, features: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess features for model input.
        
        Args:
            features (List[Dict[str, Any]]): List of feature dictionaries
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Preprocessed features and labels
        """
        return preprocess_features(features)
    
    @abstractmethod
    def train(self, features: List[Dict[str, Any]], **kwargs) -> None:
        """
        Train the model.
        
        Args:
            features (List[Dict[str, Any]]): Training features
            **kwargs: Training parameters
        """
        pass
    
    @abstractmethod
    def predict(self, features: List[Dict[str, Any]]) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            features (List[Dict[str, Any]]): Features to predict on
            
        Returns:
            np.ndarray: Predicted labels
        """
        pass
    
    @abstractmethod
    def save(self, filepath: str) -> None:
        """
        Save model weights.
        
        Args:
            filepath (str): Path to save the model
        """
        pass
    
    @abstractmethod
    def load(self, filepath: str) -> None:
        """
        Load model weights.
        
        Args:
            filepath (str): Path to load the model from
        """
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {}
    
    def set_params(self, **params) -> None:
        """Set model parameters."""
        pass
    
    @property
    def is_ready(self) -> bool:
        """Check if model is ready for inference."""
        return self.model is not None 