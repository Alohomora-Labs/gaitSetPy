"""
classification: A module for training and evaluating classification models.

This module provides both the new class-based classification models and legacy function-based API.
All classification models inherit from BaseClassificationModel and are registered with the ClassificationManager.

Available Models:
- Random Forest (fully implemented)
- MLP (PyTorch) - TODO
- LSTM (PyTorch) - TODO  
- BiLSTM (PyTorch) - TODO
- GNN (PyTorch Geometric) - TODO

Utilities:
- Dataset loading and preprocessing
- Model training and evaluation
- Feature preprocessing and preparation

Maintainer: @aharshit123456
"""

# Import the new class-based classification models
from .models.random_forest import RandomForestModel

# Import legacy functions for backward compatibility
from .models.random_forest import create_random_forest_model
from .utils.preprocess import preprocess_features
from .utils.eval import evaluate_model

# Import managers
from ..core.managers import ClassificationManager

# Register all classification models with the manager
def _register_models():
    """Register all available classification models with the ClassificationManager."""
    manager = ClassificationManager()
    manager.register_model("random_forest", RandomForestModel)

# Auto-register models when module is imported
_register_models()

# Convenient access to the classification manager
def get_classification_manager():
    """Get the singleton ClassificationManager instance."""
    return ClassificationManager()

# Helper function to get available models
def get_available_models():
    """Get list of available classification model names."""
    return ClassificationManager().get_available_components()

# Helper function to train model using manager
def train_model(model_name: str, features, **kwargs):
    """
    Train a classification model using the ClassificationManager.
    
    Args:
        model_name: Name of the classification model
        features: List of feature dictionaries
        **kwargs: Additional arguments for training
        
    Returns:
        Trained model instance
    """
    return ClassificationManager().train_model(model_name, features, **kwargs)

# Helper function to make predictions using manager
def predict(model_name: str, features, **kwargs):
    """
    Make predictions using the ClassificationManager.
    
    Args:
        model_name: Name of the classification model
        features: List of feature dictionaries
        **kwargs: Additional arguments for prediction
        
    Returns:
        Predictions array
    """
    return ClassificationManager().predict(model_name, features, **kwargs)

# Helper function to evaluate model using manager
def evaluate_model_performance(model_name: str, features, **kwargs):
    """
    Evaluate a classification model using the ClassificationManager.
    
    Args:
        model_name: Name of the classification model
        features: List of feature dictionaries
        **kwargs: Additional arguments for evaluation
        
    Returns:
        Evaluation metrics dictionary
    """
    return ClassificationManager().evaluate_model(model_name, features, **kwargs)

# Convenient wrapper functions for Random Forest
def create_random_forest(n_estimators=100, random_state=42, max_depth=None):
    """
    Create a Random Forest model with specified parameters.
    
    Args:
        n_estimators: Number of trees in the forest
        random_state: Random state for reproducibility
        max_depth: Maximum depth of the tree
        
    Returns:
        RandomForestModel instance
    """
    return RandomForestModel(n_estimators=n_estimators, random_state=random_state, max_depth=max_depth)

def train_random_forest(features, **kwargs):
    """
    Train a Random Forest model on the given features.
    
    Args:
        features: List of feature dictionaries
        **kwargs: Additional arguments for training
        
    Returns:
        Trained RandomForestModel instance
    """
    model = RandomForestModel()
    model.train(features, **kwargs)
    return model

__all__ = [
    # New class-based models
    'RandomForestModel',
    # Legacy functions for backward compatibility
    'create_random_forest_model',
    'preprocess_features',
    'evaluate_model',
    # Manager functions
    'get_classification_manager',
    'get_available_models',
    'train_model',
    'predict',
    'evaluate_model_performance',
    # Convenient wrapper functions
    'create_random_forest',
    'train_random_forest'
]
