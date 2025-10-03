"""
Unit tests for PyTorch-based classification models in GaitSetPy.

This module tests the PyTorch models including BiLSTM, CNN, GNN, LSTM, and MLP models.
Tests are designed to work with and without PyTorch availability.

Maintainer: @aharshit123456
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import patch, Mock, MagicMock
from typing import List, Dict, Any

# Test PyTorch availability
try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    torch = None
    nn = None

# Import models with conditional handling
if PYTORCH_AVAILABLE:
    from gaitsetpy.classification.models.bilstm import BiLSTMModel, BiLSTMNet
    from gaitsetpy.classification.models.cnn import CNNModel, SimpleCNN
    from gaitsetpy.classification.models.gnn import GNNModel, SimpleGCN
    from gaitsetpy.classification.models.lstm import LSTMModel, LSTMNet
    from gaitsetpy.classification.models.mlp import MLPModel
else:
    # Create mock classes for testing without PyTorch
    class MockModel:
        def __init__(self, *args, **kwargs):
            pass
        def train(self, *args, **kwargs):
            pass
        def predict(self, *args, **kwargs):
            return np.array([0, 1])
        def evaluate(self, *args, **kwargs):
            return {'accuracy': 0.8}
        def save_model(self, *args, **kwargs):
            pass
        def load_model(self, *args, **kwargs):
            pass
    
    BiLSTMModel = MockModel
    CNNModel = MockModel
    GNNModel = MockModel
    LSTMModel = MockModel
    MLPModel = MockModel

from gaitsetpy.classification.utils.preprocess import preprocess_features


@pytest.fixture
def sample_features():
    """Create sample features for testing."""
    return [
        {
            'name': 'sensor1',
            'features': {
                'mean': [1.0, 2.0, 3.0],
                'std': [0.5, 0.6, 0.7],
                'rms': [1.2, 2.1, 3.1]
            },
            'annotations': [0, 1, 0]
        },
        {
            'name': 'sensor2', 
            'features': {
                'mean': [2.0, 3.0, 4.0],
                'std': [0.8, 0.9, 1.0],
                'rms': [2.2, 3.1, 4.1]
            },
            'annotations': [1, 0, 1]
        }
    ]


@pytest.fixture
def sample_adjacency_matrix():
    """Create sample adjacency matrix for GNN testing."""
    return np.array([
        [0, 1, 1, 0, 1, 0],
        [1, 0, 1, 1, 0, 1],
        [1, 1, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1],
        [1, 0, 1, 0, 0, 1],
        [0, 1, 0, 1, 1, 0]
    ])


class TestBiLSTMModel:
    """Test cases for BiLSTM model."""
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_instantiation_default(self):
        """Test BiLSTM model instantiation with default parameters."""
        model = BiLSTMModel()
        assert model.name == "bilstm"
        assert model.description == "Bidirectional LSTM classifier for gait data classification"
        assert model.config['input_size'] == 10
        assert model.config['hidden_size'] == 64
        assert model.config['num_layers'] == 1
        assert model.config['num_classes'] == 2
        assert not model.trained
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_instantiation_custom_params(self):
        """Test BiLSTM model instantiation with custom parameters."""
        model = BiLSTMModel(
            input_size=20,
            hidden_size=128,
            num_layers=2,
            num_classes=3,
            lr=0.01,
            epochs=50,
            batch_size=64
        )
        assert model.config['input_size'] == 20
        assert model.config['hidden_size'] == 128
        assert model.config['num_layers'] == 2
        assert model.config['num_classes'] == 3
        assert model.config['lr'] == 0.01
        assert model.config['epochs'] == 50
        assert model.config['batch_size'] == 64
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_train_basic(self, sample_features):
        """Test basic training of BiLSTM model."""
        model = BiLSTMModel(input_size=3, epochs=1)  # Reduced epochs for testing
        model.train(sample_features)
        assert model.trained
        assert len(model.feature_names) > 0
        assert len(model.class_names) > 0
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_train_no_validation_split(self, sample_features):
        """Test training without validation split."""
        model = BiLSTMModel(input_size=3, epochs=1)
        model.train(sample_features, validation_split=False)
        assert model.trained
        assert not hasattr(model, 'X_test')
        assert not hasattr(model, 'y_test')
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_predict_basic(self, sample_features):
        """Test basic prediction with BiLSTM model."""
        model = BiLSTMModel(input_size=3, epochs=1)
        model.train(sample_features)
        predictions = model.predict(sample_features)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) > 0
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_predict_not_trained(self, sample_features):
        """Test prediction without training."""
        model = BiLSTMModel(input_size=3)
        with pytest.raises(ValueError, match="Model must be trained before making predictions"):
            model.predict(sample_features)
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_evaluate_basic(self, sample_features):
        """Test basic evaluation of BiLSTM model."""
        model = BiLSTMModel(input_size=3, epochs=1)
        model.train(sample_features)
        metrics = model.evaluate(sample_features)
        assert 'accuracy' in metrics
        assert 'confusion_matrix' in metrics
        assert isinstance(metrics['accuracy'], float)
        assert isinstance(metrics['confusion_matrix'], list)
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_evaluate_detailed_report(self, sample_features):
        """Test evaluation with detailed report."""
        model = BiLSTMModel(input_size=3, epochs=1)
        model.train(sample_features)
        metrics = model.evaluate(sample_features, detailed_report=True)
        assert 'classification_report' in metrics
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_evaluate_not_trained(self, sample_features):
        """Test evaluation without training."""
        model = BiLSTMModel(input_size=3)
        with pytest.raises(ValueError, match="Model must be trained before evaluation"):
            model.evaluate(sample_features)
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_save_model(self, sample_features):
        """Test saving BiLSTM model."""
        model = BiLSTMModel(input_size=3, epochs=1)
        model.train(sample_features)
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            model.save_model(tmp_path)
            assert os.path.exists(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_save_model_not_trained(self):
        """Test saving untrained model."""
        model = BiLSTMModel(input_size=3)
        with pytest.raises(ValueError, match="Model must be trained before saving"):
            model.save_model("dummy.pth")
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_load_model(self, sample_features):
        """Test loading BiLSTM model."""
        model = BiLSTMModel(input_size=3, epochs=1)
        model.train(sample_features)
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            model.save_model(tmp_path)
            
            # Create new model and load
            new_model = BiLSTMModel(input_size=3)
            new_model.load_model(tmp_path)
            assert new_model.trained
            assert new_model.config == model.config
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestCNNModel:
    """Test cases for CNN model."""
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_instantiation_default(self):
        """Test CNN model instantiation with default parameters."""
        model = CNNModel()
        assert model.name == "cnn"
        assert model.description == "1D CNN classifier for gait data classification"
        assert model.config['input_channels'] == 10
        assert model.config['num_classes'] == 2
        assert not model.trained
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_instantiation_custom_params(self):
        """Test CNN model instantiation with custom parameters."""
        model = CNNModel(
            input_channels=20,
            num_classes=3,
            lr=0.01,
            epochs=50,
            batch_size=64
        )
        assert model.config['input_channels'] == 20
        assert model.config['num_classes'] == 3
        assert model.config['lr'] == 0.01
        assert model.config['epochs'] == 50
        assert model.config['batch_size'] == 64
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_train_basic(self, sample_features):
        """Test basic training of CNN model."""
        model = CNNModel(input_channels=3, epochs=1)
        model.train(sample_features)
        assert model.trained
        assert len(model.feature_names) > 0
        assert len(model.class_names) > 0
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_predict_basic(self, sample_features):
        """Test basic prediction with CNN model."""
        model = CNNModel(input_channels=3, epochs=1)
        model.train(sample_features)
        predictions = model.predict(sample_features)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) > 0
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_evaluate_basic(self, sample_features):
        """Test basic evaluation of CNN model."""
        model = CNNModel(input_channels=3, epochs=1)
        model.train(sample_features)
        metrics = model.evaluate(sample_features)
        assert 'accuracy' in metrics
        assert 'confusion_matrix' in metrics


class TestGNNModel:
    """Test cases for GNN model."""
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_instantiation_default(self):
        """Test GNN model instantiation with default parameters."""
        model = GNNModel()
        assert model.name == "gnn"
        assert model.description == "Graph Convolutional Network (GCN) classifier for gait data classification"
        assert model.config['input_dim'] == 10
        assert model.config['hidden_dim'] == 32
        assert model.config['output_dim'] == 2
        assert not model.trained
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_instantiation_custom_params(self):
        """Test GNN model instantiation with custom parameters."""
        model = GNNModel(
            input_dim=20,
            hidden_dim=64,
            output_dim=3,
            lr=0.01,
            epochs=50
        )
        assert model.config['input_dim'] == 20
        assert model.config['hidden_dim'] == 64
        assert model.config['output_dim'] == 3
        assert model.config['lr'] == 0.01
        assert model.config['epochs'] == 50
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_train_basic(self, sample_features, sample_adjacency_matrix):
        """Test basic training of GNN model."""
        model = GNNModel(input_dim=3, epochs=1)
        model.train(sample_features, adjacency_matrix=sample_adjacency_matrix)
        assert model.trained
        assert len(model.feature_names) > 0
        assert len(model.class_names) > 0
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_train_missing_adjacency_matrix(self, sample_features):
        """Test training without adjacency matrix."""
        model = GNNModel(input_dim=3, epochs=1)
        with pytest.raises(ValueError, match="Adjacency matrix must be provided"):
            model.train(sample_features)
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_predict_basic(self, sample_features, sample_adjacency_matrix):
        """Test basic prediction with GNN model."""
        model = GNNModel(input_dim=3, epochs=1)
        model.train(sample_features, adjacency_matrix=sample_adjacency_matrix)
        predictions = model.predict(sample_features, adjacency_matrix=sample_adjacency_matrix)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) > 0
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_predict_missing_adjacency_matrix(self, sample_features, sample_adjacency_matrix):
        """Test prediction without adjacency matrix."""
        model = GNNModel(input_dim=3, epochs=1)
        model.train(sample_features, adjacency_matrix=sample_adjacency_matrix)
        with pytest.raises(ValueError, match="Adjacency matrix must be provided"):
            model.predict(sample_features)
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_evaluate_basic(self, sample_features, sample_adjacency_matrix):
        """Test basic evaluation of GNN model."""
        model = GNNModel(input_dim=3, epochs=1)
        model.train(sample_features, adjacency_matrix=sample_adjacency_matrix)
        metrics = model.evaluate(sample_features, adjacency_matrix=sample_adjacency_matrix)
        assert 'accuracy' in metrics
        assert 'confusion_matrix' in metrics


class TestLSTMModel:
    """Test cases for LSTM model."""
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_instantiation_default(self):
        """Test LSTM model instantiation with default parameters."""
        model = LSTMModel()
        assert model.name == "lstm"
        assert model.description == "LSTM classifier for gait data classification"
        assert model.config['input_size'] == 10
        assert model.config['hidden_size'] == 64
        assert model.config['num_layers'] == 1
        assert model.config['num_classes'] == 2
        assert not model.trained
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_instantiation_custom_params(self):
        """Test LSTM model instantiation with custom parameters."""
        model = LSTMModel(
            input_size=20,
            hidden_size=128,
            num_layers=2,
            num_classes=3,
            lr=0.01,
            epochs=50,
            batch_size=64
        )
        assert model.config['input_size'] == 20
        assert model.config['hidden_size'] == 128
        assert model.config['num_layers'] == 2
        assert model.config['num_classes'] == 3
        assert model.config['lr'] == 0.01
        assert model.config['epochs'] == 50
        assert model.config['batch_size'] == 64
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_train_basic(self, sample_features):
        """Test basic training of LSTM model."""
        model = LSTMModel(input_size=3, epochs=1)
        model.train(sample_features)
        assert model.trained
        assert len(model.feature_names) > 0
        assert len(model.class_names) > 0
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_predict_basic(self, sample_features):
        """Test basic prediction with LSTM model."""
        model = LSTMModel(input_size=3, epochs=1)
        model.train(sample_features)
        predictions = model.predict(sample_features)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) > 0
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_evaluate_basic(self, sample_features):
        """Test basic evaluation of LSTM model."""
        model = LSTMModel(input_size=3, epochs=1)
        model.train(sample_features)
        metrics = model.evaluate(sample_features)
        assert 'accuracy' in metrics
        assert 'confusion_matrix' in metrics


class TestMLPModel:
    """Test cases for MLP model."""
    
    def test_instantiation_default(self):
        """Test MLP model instantiation with default parameters."""
        model = MLPModel()
        assert model.name == "mlp"
        assert model.description == "Multi-Layer Perceptron classifier for gait data classification"
        assert model.config['hidden_layer_sizes'] == (100,)
        assert model.config['activation'] == 'relu'
        assert model.config['solver'] == 'adam'
        assert model.config['random_state'] == 42
        assert model.config['max_iter'] == 200
        assert not model.trained
    
    def test_instantiation_custom_params(self):
        """Test MLP model instantiation with custom parameters."""
        model = MLPModel(
            hidden_layer_sizes=(200, 100),
            activation='tanh',
            solver='lbfgs',
            random_state=123,
            max_iter=500
        )
        assert model.config['hidden_layer_sizes'] == (200, 100)
        assert model.config['activation'] == 'tanh'
        assert model.config['solver'] == 'lbfgs'
        assert model.config['random_state'] == 123
        assert model.config['max_iter'] == 500
    
    def test_train_basic(self, sample_features):
        """Test basic training of MLP model."""
        model = MLPModel(max_iter=10)  # Reduced iterations for testing
        model.train(sample_features)
        assert model.trained
        assert len(model.feature_names) > 0
        assert len(model.class_names) > 0
    
    def test_train_no_validation_split(self, sample_features):
        """Test training without validation split."""
        model = MLPModel(max_iter=10)
        model.train(sample_features, validation_split=False)
        assert model.trained
        assert not hasattr(model, 'X_test')
        assert not hasattr(model, 'y_test')
    
    def test_predict_basic(self, sample_features):
        """Test basic prediction with MLP model."""
        model = MLPModel(max_iter=10)
        model.train(sample_features)
        predictions = model.predict(sample_features)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) > 0
    
    def test_predict_probabilities(self, sample_features):
        """Test prediction with probabilities."""
        model = MLPModel(max_iter=10)
        model.train(sample_features)
        probabilities = model.predict(sample_features, return_probabilities=True)
        assert isinstance(probabilities, np.ndarray)
        assert len(probabilities) > 0
        # Check that probabilities sum to 1 for each sample
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-10)
    
    def test_predict_not_trained(self, sample_features):
        """Test prediction without training."""
        model = MLPModel()
        with pytest.raises(ValueError, match="Model must be trained before making predictions"):
            model.predict(sample_features)
    
    def test_evaluate_basic(self, sample_features):
        """Test basic evaluation of MLP model."""
        model = MLPModel(max_iter=10)
        model.train(sample_features)
        metrics = model.evaluate(sample_features)
        assert 'accuracy' in metrics
        assert 'confusion_matrix' in metrics
        assert isinstance(metrics['accuracy'], float)
        assert isinstance(metrics['confusion_matrix'], list)
    
    def test_evaluate_detailed_report(self, sample_features):
        """Test evaluation with detailed report."""
        model = MLPModel(max_iter=10)
        model.train(sample_features)
        metrics = model.evaluate(sample_features, detailed_report=True)
        assert 'classification_report' in metrics
    
    def test_evaluate_not_trained(self, sample_features):
        """Test evaluation without training."""
        model = MLPModel()
        with pytest.raises(ValueError, match="Model must be trained before evaluation"):
            model.evaluate(sample_features)
    
    def test_save_model(self, sample_features):
        """Test saving MLP model."""
        model = MLPModel(max_iter=10)
        model.train(sample_features)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            model.save_model(tmp_path)
            assert os.path.exists(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_save_model_not_trained(self):
        """Test saving untrained model."""
        model = MLPModel()
        with pytest.raises(ValueError, match="Model must be trained before saving"):
            model.save_model("dummy.pkl")
    
    def test_load_model_dict_format(self, sample_features):
        """Test loading MLP model in dict format."""
        model = MLPModel(max_iter=10)
        model.train(sample_features)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            model.save_model(tmp_path)
            
            # Create new model and load
            new_model = MLPModel()
            new_model.load_model(tmp_path)
            assert new_model.trained
            assert new_model.config == model.config
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_load_model_legacy_format(self):
        """Test loading MLP model in legacy format."""
        # Create a mock legacy model file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save a simple model object (legacy format)
            import joblib
            from sklearn.neural_network import MLPClassifier
            legacy_model = MLPClassifier(max_iter=10)
            joblib.dump(legacy_model, tmp_path)
            
            # Load with our model
            model = MLPModel()
            model.load_model(tmp_path)
            assert model.trained
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_load_model_error(self):
        """Test loading model with error handling."""
        model = MLPModel()
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Write invalid data
            with open(tmp_path, 'w') as f:
                f.write("invalid data")
            
            # Should handle error gracefully
            model.load_model(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestPyTorchModelEdgeCases:
    """Test edge cases for PyTorch models."""
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_train_empty_features(self):
        """Test training with empty features."""
        model = BiLSTMModel(input_size=3, epochs=1)
        with pytest.raises(ValueError):
            model.train([])
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_train_single_class(self):
        """Test training with single class."""
        features = [
            {
                'name': 'sensor1',
                'features': {'mean': [1.0, 2.0], 'std': [0.5, 0.6]},
                'annotations': [0, 0]  # Single class
            }
        ]
        model = BiLSTMModel(input_size=2, epochs=1)
        # Should handle single class gracefully
        model.train(features)
        assert model.trained
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_train_single_sample(self):
        """Test training with single sample."""
        features = [
            {
                'name': 'sensor1',
                'features': {'mean': [1.0], 'std': [0.5]},
                'annotations': [0]
            }
        ]
        model = BiLSTMModel(input_size=2, epochs=1)
        # Should handle single sample gracefully by disabling validation split
        model.train(features, validation_split=False)
        assert model.trained
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_predict_empty_features(self):
        """Test prediction with empty features."""
        model = BiLSTMModel(input_size=2, epochs=1)
        # Train with some data first
        sample_features = [
            {
                'name': 'sensor1',
                'features': {'mean': [1.0, 2.0], 'std': [0.5, 0.6]},
                'annotations': [0, 1]
            }
        ]
        model.train(sample_features)
        
        # Now test with empty features - should raise ValueError
        with pytest.raises(ValueError, match="No valid features or labels found"):
            model.predict([])


class TestPyTorchModelIntegration:
    """Integration tests for PyTorch models."""
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_full_workflow_bilstm(self, sample_features):
        """Test complete workflow for BiLSTM model."""
        model = BiLSTMModel(input_size=3, epochs=1)
        
        # Train
        model.train(sample_features)
        assert model.trained
        
        # Predict
        predictions = model.predict(sample_features)
        assert len(predictions) > 0
        
        # Evaluate
        metrics = model.evaluate(sample_features)
        assert 'accuracy' in metrics
        
        # Save and load
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            model.save_model(tmp_path)
            new_model = BiLSTMModel(input_size=3)
            new_model.load_model(tmp_path)
            assert new_model.trained
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_full_workflow_mlp(self, sample_features):
        """Test complete workflow for MLP model."""
        model = MLPModel(max_iter=10)
        
        # Train
        model.train(sample_features)
        assert model.trained
        
        # Predict
        predictions = model.predict(sample_features)
        assert len(predictions) > 0
        
        # Predict with probabilities
        probabilities = model.predict(sample_features, return_probabilities=True)
        assert len(probabilities) > 0
        
        # Evaluate
        metrics = model.evaluate(sample_features)
        assert 'accuracy' in metrics
        
        # Save and load
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            model.save_model(tmp_path)
            new_model = MLPModel()
            new_model.load_model(tmp_path)
            assert new_model.trained
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestPyTorchAvailability:
    """Test PyTorch availability and fallback behavior."""
    
    def test_pytorch_availability_flag(self):
        """Test that PyTorch availability is properly detected."""
        # This test will pass regardless of PyTorch availability
        assert isinstance(PYTORCH_AVAILABLE, bool)
    
    @pytest.mark.skipif(PYTORCH_AVAILABLE, reason="PyTorch is available")
    def test_models_without_pytorch(self, sample_features):
        """Test that models work with mock implementations when PyTorch is not available."""
        # These should work with mock implementations
        models = [BiLSTMModel(), CNNModel(), GNNModel(), LSTMModel(), MLPModel()]
        
        for model in models:
            # Basic instantiation should work
            assert hasattr(model, 'train')
            assert hasattr(model, 'predict')
            assert hasattr(model, 'evaluate')
            
            # Mock methods should return expected types
            predictions = model.predict(sample_features)
            assert isinstance(predictions, np.ndarray)
            
            metrics = model.evaluate(sample_features)
            assert isinstance(metrics, dict)
            assert 'accuracy' in metrics
