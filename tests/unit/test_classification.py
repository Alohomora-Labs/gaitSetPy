"""
Unit tests for classification functionality in GaitSetPy.

This module tests the classification models and utilities
including Random Forest and other ML models.

Maintainer: @aharshit123456
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, Mock, MagicMock
import joblib
import tempfile
import os

from gaitsetpy.classification.models.random_forest import RandomForestModel
from gaitsetpy.classification.utils.preprocess import preprocess_features


class TestPreprocessFeatures:
    """Test cases for feature preprocessing utilities."""
    
    def test_preprocess_features_basic(self, sample_features):
        """Test basic feature preprocessing."""
        # Create properly formatted features with annotations for each sensor
        formatted_features = []
        annotations = [1, 2, 1, 2, 1]
        
        for sensor_data in sample_features[:-1]:  # Exclude the annotations entry
            formatted_features.append({
                'name': sensor_data['name'],
                'features': sensor_data['features'],
                'annotations': annotations
            })
        
        X, y = preprocess_features(formatted_features)
        
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert len(X) == len(y)
        assert X.shape[1] > 0  # Should have features
    
    def test_preprocess_features_empty(self):
        """Test preprocessing with empty features."""
        with pytest.raises(ValueError, match="No valid features or labels found"):
            X, y = preprocess_features([])
    
    def test_preprocess_features_single_sensor(self):
        """Test preprocessing with single sensor."""
        features = [{
            'name': 'sensor1',
            'features': {
                'mean': [1, 2, 3],
                'std': [0.1, 0.2, 0.3]
            },
            'annotations': [1, 2, 1]
        }]
        
        X, y = preprocess_features(features)
        
        assert X.shape == (3, 2)  # 3 samples, 2 features
        assert len(y) == 3
        # Labels are standardized to start from 0, so [1, 2, 1] becomes [0, 1, 0]
        assert np.array_equal(y, [0, 1, 0])
    
    def test_preprocess_features_multiple_sensors(self):
        """Test preprocessing with multiple sensors."""
        features = [
            {
                'name': 'sensor1',
                'features': {
                    'mean': [1, 2],
                    'std': [0.1, 0.2]
                },
                'annotations': [1, 2]
            },
            {
                'name': 'sensor2',
                'features': {
                    'mean': [3, 4],
                    'std': [0.3, 0.4]
                },
                'annotations': [1, 2]
            }
        ]
        
        X, y = preprocess_features(features)
        
        assert X.shape == (4, 2)  # 4 samples (2 per sensor), 2 features
        assert len(y) == 4
    
    def test_preprocess_features_mismatched_lengths(self):
        """Test preprocessing with mismatched feature lengths."""
        features = [{
            'name': 'sensor1',
            'features': {
                'mean': [1, 2, 3],
                'std': [0.1, 0.2]  # Different length
            },
            'annotations': [1, 2, 1]
        }]
        
        # Should handle gracefully
        X, y = preprocess_features(features)
        assert len(X) > 0
        assert len(y) > 0


class TestRandomForestModel:
    """Test cases for RandomForestModel."""
    
    def test_instantiation_default(self):
        """Test RandomForestModel instantiation with default parameters."""
        model = RandomForestModel()
        
        assert model.name == "random_forest"
        assert model.config['n_estimators'] == 100
        assert model.config['random_state'] == 42
        assert model.config['max_depth'] is None
        assert model.trained is False
        assert model.model is not None
    
    def test_instantiation_custom_params(self):
        """Test RandomForestModel instantiation with custom parameters."""
        model = RandomForestModel(n_estimators=50, random_state=123, max_depth=10)
        
        assert model.config['n_estimators'] == 50
        assert model.config['random_state'] == 123
        assert model.config['max_depth'] == 10
    
    def test_train_basic(self, sample_features):
        """Test basic model training."""
        model = RandomForestModel(n_estimators=10, random_state=42)
        
        with patch('gaitsetpy.classification.models.random_forest.preprocess_features') as mock_preprocess:
            mock_preprocess.return_value = (
                np.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
                np.array([1, 2, 1, 2])
            )
            
            model.train(sample_features, test_size=0.2, validation_split=True)
            
            assert model.trained is True
            assert hasattr(model, 'X_test')
            assert hasattr(model, 'y_test')
            mock_preprocess.assert_called_once_with(sample_features)
    
    def test_train_no_validation_split(self, sample_features):
        """Test model training without validation split."""
        model = RandomForestModel(n_estimators=10, random_state=42)
        
        with patch('gaitsetpy.classification.models.random_forest.preprocess_features') as mock_preprocess:
            mock_preprocess.return_value = (
                np.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
                np.array([1, 2, 1, 2])
            )
            
            model.train(sample_features, validation_split=False)
            
            assert model.trained is True
            assert not hasattr(model, 'X_test')
            assert not hasattr(model, 'y_test')
    
    def test_predict_basic(self, sample_features):
        """Test basic prediction."""
        model = RandomForestModel(n_estimators=10, random_state=42)
        
        # Train the model first
        with patch('gaitsetpy.classification.models.random_forest.preprocess_features') as mock_preprocess:
            mock_preprocess.return_value = (
                np.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
                np.array([1, 2, 1, 2])
            )
            model.train(sample_features, validation_split=False)
        
        # Test prediction
        with patch('gaitsetpy.classification.models.random_forest.preprocess_features') as mock_preprocess:
            mock_preprocess.return_value = (
                np.array([[1, 2], [3, 4]]),
                np.array([1, 2])
            )
            
            predictions = model.predict(sample_features)
            
            assert isinstance(predictions, np.ndarray)
            assert len(predictions) == 2
    
    def test_predict_probabilities(self, sample_features):
        """Test prediction with probabilities."""
        model = RandomForestModel(n_estimators=10, random_state=42)
        
        # Train the model first
        with patch('gaitsetpy.classification.models.random_forest.preprocess_features') as mock_preprocess:
            mock_preprocess.return_value = (
                np.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
                np.array([1, 2, 1, 2])
            )
            model.train(sample_features, validation_split=False)
        
        # Test prediction with probabilities
        with patch('gaitsetpy.classification.models.random_forest.preprocess_features') as mock_preprocess:
            mock_preprocess.return_value = (
                np.array([[1, 2], [3, 4]]),
                np.array([1, 2])
            )
            
            probabilities = model.predict(sample_features, return_probabilities=True)
            
            assert isinstance(probabilities, np.ndarray)
            assert probabilities.shape[0] == 2
            assert probabilities.shape[1] == 2  # 2 classes
    
    def test_predict_not_trained(self, sample_features):
        """Test prediction without training."""
        model = RandomForestModel()
        
        with pytest.raises(ValueError, match="Model must be trained before making predictions"):
            model.predict(sample_features)
    
    def test_evaluate_basic(self, sample_features):
        """Test basic model evaluation."""
        model = RandomForestModel(n_estimators=10, random_state=42)
        
        # Train the model first
        with patch('gaitsetpy.classification.models.random_forest.preprocess_features') as mock_preprocess:
            mock_preprocess.return_value = (
                np.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
                np.array([1, 2, 1, 2])
            )
            model.train(sample_features, test_size=0.2, validation_split=True)
        
        # Test evaluation
        metrics = model.evaluate(sample_features)
        
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'confusion_matrix' in metrics
        assert isinstance(metrics['accuracy'], float)
        assert isinstance(metrics['confusion_matrix'], list)
    
    def test_evaluate_detailed_report(self, sample_features):
        """Test evaluation with detailed report."""
        model = RandomForestModel(n_estimators=10, random_state=42)
        
        # Train the model first
        with patch('gaitsetpy.classification.models.random_forest.preprocess_features') as mock_preprocess:
            mock_preprocess.return_value = (
                np.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
                np.array([1, 2, 1, 2])
            )
            model.train(sample_features, test_size=0.2, validation_split=True)
        
        # Test evaluation with detailed report
        metrics = model.evaluate(sample_features, detailed_report=True)
        
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'confusion_matrix' in metrics
        assert 'classification_report' in metrics
        assert 'feature_importance' in metrics
    
    def test_evaluate_not_trained(self, sample_features):
        """Test evaluation without training."""
        model = RandomForestModel()
        
        with pytest.raises(ValueError, match="Model must be trained before evaluation"):
            model.evaluate(sample_features)
    
    def test_save_model(self, sample_features):
        """Test saving a trained model."""
        model = RandomForestModel(n_estimators=10, random_state=42)
        
        # Train the model first
        with patch('gaitsetpy.classification.models.random_forest.preprocess_features') as mock_preprocess:
            mock_preprocess.return_value = (
                np.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
                np.array([1, 2, 1, 2])
            )
            model.train(sample_features, validation_split=False)
        
        # Test saving
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            with patch('joblib.dump') as mock_dump:
                model.save_model(tmp_path)
                mock_dump.assert_called_once()
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_save_model_not_trained(self):
        """Test saving model without training."""
        model = RandomForestModel()
        
        with pytest.raises(ValueError, match="Model must be trained before saving"):
            model.save_model("dummy_path.pkl")
    
    def test_load_model_dict_format(self):
        """Test loading model with dictionary format."""
        model = RandomForestModel()
        
        # Mock model data
        mock_model_data = {
            'model': Mock(),
            'config': {'n_estimators': 50, 'random_state': 42},
            'feature_names': ['feature1', 'feature2'],
            'class_names': [1, 2],
            'trained': True
        }
        
        with patch('joblib.load', return_value=mock_model_data):
            model.load_model("dummy_path.pkl")
            
            assert model.model == mock_model_data['model']
            assert model.config == mock_model_data['config']
            assert model.feature_names == mock_model_data['feature_names']
            assert model.class_names == mock_model_data['class_names']
            assert model.trained == mock_model_data['trained']
    
    def test_load_model_legacy_format(self):
        """Test loading model with legacy format."""
        model = RandomForestModel()
        
        # Mock legacy model data (just the model)
        mock_model = Mock()
        
        with patch('joblib.load', return_value=mock_model):
            model.load_model("dummy_path.pkl")
            
            assert model.model == mock_model
            assert model.trained is True
    
    def test_load_model_error(self):
        """Test loading model with error."""
        model = RandomForestModel()
        
        with patch('joblib.load', side_effect=Exception("Load error")):
            with pytest.raises(Exception, match="Load error"):
                model.load_model("dummy_path.pkl")
    
    def test_get_feature_importance(self, sample_features):
        """Test getting feature importance."""
        model = RandomForestModel(n_estimators=10, random_state=42)
        
        # Train the model first
        with patch('gaitsetpy.classification.models.random_forest.preprocess_features') as mock_preprocess:
            mock_preprocess.return_value = (
                np.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
                np.array([1, 2, 1, 2])
            )
            model.train(sample_features, validation_split=False)
        
        # The model should have feature_importances_ after training
        model.feature_names = ['feature1', 'feature2']
        
        importance = model.get_feature_importance()
        
        assert isinstance(importance, dict)
        assert 'feature1' in importance
        assert 'feature2' in importance
        # Check that importance values are between 0 and 1
        assert all(0 <= val <= 1 for val in importance.values)
    
    def test_get_feature_importance_not_trained(self):
        """Test getting feature importance without training."""
        model = RandomForestModel()
        
        with pytest.raises(ValueError, match="Model must be trained to get feature importance"):
            model.get_feature_importance()
    
    def test_predict_single(self, sample_features):
        """Test predicting single feature vector."""
        model = RandomForestModel(n_estimators=10, random_state=42)
        
        # Train the model first
        with patch('gaitsetpy.classification.models.random_forest.preprocess_features') as mock_preprocess:
            mock_preprocess.return_value = (
                np.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
                np.array([1, 2, 1, 2])
            )
            model.train(sample_features, validation_split=False)
        
        # Test single prediction
        single_features = {
            'name': 'sensor1',
            'features': {
                'mean': [1],
                'std': [0.1]
            },
            'annotations': [1]
        }
        
        with patch('gaitsetpy.classification.models.random_forest.preprocess_features') as mock_preprocess:
            mock_preprocess.return_value = (
                np.array([[1, 0.1]]),
                np.array([1])
            )
            
            prediction = model.predict_single(single_features)
            
            assert isinstance(prediction, (int, np.integer))
    
    def test_predict_single_not_trained(self):
        """Test single prediction without training."""
        model = RandomForestModel()
        
        with pytest.raises(ValueError, match="Model must be trained before making predictions"):
            model.predict_single({})


class TestRandomForestModelEdgeCases:
    """Test edge cases for RandomForestModel."""
    
    def test_train_empty_features(self):
        """Test training with empty features."""
        model = RandomForestModel(n_estimators=10, random_state=42)
        
        with patch('gaitsetpy.classification.models.random_forest.preprocess_features') as mock_preprocess:
            mock_preprocess.return_value = (np.array([]), np.array([]))
            
            # Should raise an error for empty data
            with pytest.raises((ValueError, IndexError)):
                model.train([], validation_split=False)
    
    def test_train_single_class(self):
        """Test training with single class."""
        model = RandomForestModel(n_estimators=10, random_state=42)
        
        with patch('gaitsetpy.classification.models.random_forest.preprocess_features') as mock_preprocess:
            mock_preprocess.return_value = (
                np.array([[1, 2], [3, 4]]),
                np.array([1, 1])  # Single class
            )
            
            model.train([], validation_split=False)
            assert model.trained is True
    
    def test_train_single_sample(self):
        """Test training with single sample."""
        model = RandomForestModel(n_estimators=10, random_state=42)
        
        with patch('gaitsetpy.classification.models.random_forest.preprocess_features') as mock_preprocess:
            mock_preprocess.return_value = (
                np.array([[1, 2]]),
                np.array([1])
            )
            
            # Should handle single sample gracefully
            model.train([], validation_split=False)
            assert model.trained is True
    
    def test_predict_empty_features(self):
        """Test prediction with empty features."""
        model = RandomForestModel(n_estimators=10, random_state=42)
        
        # Train the model first
        with patch('gaitsetpy.classification.models.random_forest.preprocess_features') as mock_preprocess:
            mock_preprocess.return_value = (
                np.array([[1, 2], [3, 4]]),
                np.array([1, 2])
            )
            model.train([], validation_split=False)
        
        # Test prediction with empty features
        with patch('gaitsetpy.classification.models.random_forest.preprocess_features') as mock_preprocess:
            mock_preprocess.return_value = (np.array([]), np.array([]))
            
            # Should raise an error for empty prediction data
            with pytest.raises(ValueError):
                predictions = model.predict([])


class TestRandomForestModelIntegration:
    """Integration tests for RandomForestModel."""
    
    def test_full_workflow(self):
        """Test complete workflow from training to prediction."""
        # Create sample data
        features = [
            {
                'name': 'sensor1',
                'features': {
                    'mean': [1, 2, 3, 4, 5],
                    'std': [0.1, 0.2, 0.3, 0.4, 0.5]
                },
                'annotations': [1, 2, 1, 2, 1]
            }
        ]
        
        model = RandomForestModel(n_estimators=10, random_state=42)
        
        # Train
        model.train(features, test_size=0.2, validation_split=True)
        assert model.trained is True
        
        # Evaluate
        metrics = model.evaluate(features, detailed_report=True)
        assert 'accuracy' in metrics
        assert 'confusion_matrix' in metrics
        
        # Predict
        predictions = model.predict(features)
        assert len(predictions) == 5
        
        # Get feature importance
        importance = model.get_feature_importance()
        assert isinstance(importance, dict)
        
        # Save and load
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            model.save_model(tmp_path)
            
            # Create new model and load
            new_model = RandomForestModel()
            new_model.load_model(tmp_path)
            
            assert new_model.trained is True
            assert new_model.config == model.config
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
