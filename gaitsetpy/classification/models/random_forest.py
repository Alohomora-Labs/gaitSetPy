'''
Random Forest Classification Model

This module contains the RandomForestModel class which inherits from BaseClassificationModel
and provides Random Forest classification functionality.

Maintainer: @aharshit123456
'''

import joblib
import numpy as np
from typing import List, Dict, Any, Optional, Union
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from ...core.base_classes import BaseClassificationModel
from ..utils.preprocess import preprocess_features


class RandomForestModel(BaseClassificationModel):
    """
    Random Forest classification model.
    
    This class provides Random Forest classification functionality with
    comprehensive training, prediction, and evaluation capabilities.
    """
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42, max_depth: Optional[int] = None):
        super().__init__(
            name="random_forest",
            description="Random Forest classifier for gait data classification"
        )
        self.config = {
            'n_estimators': n_estimators,
            'random_state': random_state,
            'max_depth': max_depth
        }
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=max_depth
        )
        self.feature_names = []
        self.class_names = []
        
    def train(self, features: List[Dict], **kwargs):
        """
        Train the Random Forest model on the given features.
        
        Args:
            features: List of feature dictionaries
            **kwargs: Additional arguments including test_size, validation_split
        """
        # Preprocess features
        X, y = preprocess_features(features)
        
        # Store feature and class information
        self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        self.class_names = list(set(y))
        
        # Split data if test_size is specified
        test_size = kwargs.get('test_size', 0.2)
        validation_split = kwargs.get('validation_split', True)
        
        if validation_split:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.config['random_state']
            )
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Store validation data for later evaluation
            self.X_test = X_test
            self.y_test = y_test
            
            # Print training accuracy
            train_accuracy = self.model.score(X_train, y_train)
            test_accuracy = self.model.score(X_test, y_test)
            
            print(f"Training accuracy: {train_accuracy:.4f}")
            print(f"Validation accuracy: {test_accuracy:.4f}")
        else:
            # Train on all data
            self.model.fit(X, y)
            train_accuracy = self.model.score(X, y)
            print(f"Training accuracy: {train_accuracy:.4f}")
        
        self.trained = True
        print("Random Forest model trained successfully.")
    
    def predict(self, features: List[Dict], **kwargs) -> Union[np.ndarray, Any]:
        """
        Make predictions using the trained Random Forest model.
        
        Args:
            features: List of feature dictionaries
            **kwargs: Additional arguments including return_probabilities
            
        Returns:
            Array of predictions or probabilities
        """
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Preprocess features
        X, _ = preprocess_features(features)
        
        # Make predictions
        return_probabilities = kwargs.get('return_probabilities', False)
        
        if return_probabilities:
            return self.model.predict_proba(X)
        else:
            return self.model.predict(X)
    
    def evaluate(self, features: List[Dict], **kwargs) -> Dict[str, float]:
        """
        Evaluate the Random Forest model performance.
        
        Args:
            features: List of feature dictionaries
            **kwargs: Additional arguments including detailed_report
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Use validation data if available, otherwise use provided features
        if hasattr(self, 'X_test') and hasattr(self, 'y_test'):
            X_test, y_test = self.X_test, self.y_test
        else:
            X_test, y_test = preprocess_features(features)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Basic metrics
        metrics = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix.tolist()
        }
        
        # Detailed report if requested
        detailed_report = kwargs.get('detailed_report', False)
        if detailed_report:
            class_report = classification_report(y_test, y_pred, output_dict=True)
            metrics['classification_report'] = class_report
            
            # Feature importance
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
                metrics['feature_importance'] = feature_importance
        
        return metrics
    
    def save_model(self, filepath: str):
        """
        Save the trained Random Forest model to a file.
        
        Args:
            filepath: Path to save the model
        """
        if not self.trained:
            raise ValueError("Model must be trained before saving")
        
        # Save model with additional metadata
        model_data = {
            'model': self.model,
            'config': self.config,
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'trained': self.trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"Random Forest model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained Random Forest model from a file.
        
        Args:
            filepath: Path to the saved model
        """
        try:
            model_data = joblib.load(filepath)
            
            # Handle legacy model format
            if isinstance(model_data, dict):
                self.model = model_data['model']
                self.config = model_data.get('config', self.config)
                self.feature_names = model_data.get('feature_names', [])
                self.class_names = model_data.get('class_names', [])
                self.trained = model_data.get('trained', True)
            else:
                # Legacy format - just the model
                self.model = model_data
                self.trained = True
            
            print(f"Random Forest model loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.trained:
            raise ValueError("Model must be trained to get feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.feature_names, self.model.feature_importances_))
        else:
            return {}
    
    def predict_single(self, single_features: Dict) -> int:
        """
        Make prediction for a single feature vector.
        
        Args:
            single_features: Dictionary containing features for a single sample
            
        Returns:
            Predicted class
        """
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Convert single feature dict to format expected by preprocess_features
        features_list = [single_features]
        X, _ = preprocess_features(features_list)
        
        return self.model.predict(X)[0]


# Legacy function wrapper for backward compatibility
def create_random_forest_model(n_estimators=100, random_state=42, max_depth=None):
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
