"""
Random Forest Model Implementation

This module provides a Random Forest classifier implementation that inherits from BaseModel.
"""

from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from .base import BaseModel

class RandomForestModel(BaseModel):
    """Random Forest classifier implementation."""
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        """
        Initialize the Random Forest model.
        
        Args:
            n_estimators (int): Number of trees in the forest
            random_state (int): Random state for reproducibility
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.random_state = random_state
        
    def build(self, **kwargs) -> None:
        """Build the Random Forest model."""
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            **kwargs
        )
        
    def train(self, features: List[Dict[str, Any]], **kwargs) -> None:
        """
        Train the Random Forest model.
        
        Args:
            features (List[Dict[str, Any]]): Training features
            **kwargs: Additional training parameters
        """
        if self.model is None:
            self.build()
            
        X, y = self.preprocess_features(features)
        self.model.fit(X, y)
        self.is_trained = True
        print("Model trained successfully.")
        
    def predict(self, features: List[Dict[str, Any]]) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            features (List[Dict[str, Any]]): Features to predict on
            
        Returns:
            np.ndarray: Predicted labels
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions.")
            
        X, _ = self.preprocess_features(features)
        return self.model.predict(X)
    
    def save(self, filepath: str) -> None:
        """
        Save the trained model to a file.
        
        Args:
            filepath (str): Path to save the model
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model.")
            
        joblib.dump(self.model, filepath)
        print("Model saved successfully.")
        
    def load(self, filepath: str) -> None:
        """
        Load a trained model from a file.
        
        Args:
            filepath (str): Path to load the model from
        """
        if filepath == "random_forest_model.pkl":
            filepath = "../weights/random_forest_model.pkl"
            
        self.model = joblib.load(filepath)
        self.is_trained = True
        print("Model loaded successfully.")
        
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            "n_estimators": self.n_estimators,
            "random_state": self.random_state
        }
        
    def set_params(self, **params) -> None:
        """Set model parameters."""
        if "n_estimators" in params:
            self.n_estimators = params["n_estimators"]
        if "random_state" in params:
            self.random_state = params["random_state"]
            
        # Rebuild model if it exists
        if self.model is not None:
            self.build()
