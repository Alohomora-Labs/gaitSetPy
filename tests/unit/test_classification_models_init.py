"""
Unit tests for the classification models __init__.py module in GaitSetPy.

This module tests the model factory function and PyTorch availability handling
in the classification models package.

Maintainer: @aharshit123456
"""

import pytest
from unittest.mock import patch, Mock, MagicMock

# Test PyTorch availability
try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    torch = None

from gaitsetpy.classification.models import (
    get_classification_model,
    PYTORCH_AVAILABLE as MODULE_PYTORCH_AVAILABLE,
    RandomForestModel,
    MLPModel
)

# Import PyTorch models with conditional handling
if PYTORCH_AVAILABLE:
    from gaitsetpy.classification.models import (
        LSTMModel,
        BiLSTMModel,
        GNNModel,
        CNNModel
    )
else:
    LSTMModel = None
    BiLSTMModel = None
    GNNModel = None
    CNNModel = None


class TestPyTorchAvailability:
    """Test PyTorch availability detection."""
    
    def test_pytorch_availability_consistency(self):
        """Test that PyTorch availability is consistent between test and module."""
        assert PYTORCH_AVAILABLE == MODULE_PYTORCH_AVAILABLE
    
    def test_pytorch_availability_flag_type(self):
        """Test that PyTorch availability flag is boolean."""
        assert isinstance(MODULE_PYTORCH_AVAILABLE, bool)


class TestModelImports:
    """Test model imports."""
    
    def test_random_forest_import(self):
        """Test that RandomForestModel is always available."""
        assert RandomForestModel is not None
        assert hasattr(RandomForestModel, '__init__')
    
    def test_mlp_import(self):
        """Test that MLPModel is always available."""
        assert MLPModel is not None
        assert hasattr(MLPModel, '__init__')
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_pytorch_models_import_with_pytorch(self):
        """Test that PyTorch models are imported when PyTorch is available."""
        assert LSTMModel is not None
        assert BiLSTMModel is not None
        assert GNNModel is not None
        assert CNNModel is not None
        
        # Test that they have the expected methods
        for model_class in [LSTMModel, BiLSTMModel, GNNModel, CNNModel]:
            assert hasattr(model_class, '__init__')
            assert hasattr(model_class, 'train')
            assert hasattr(model_class, 'predict')
            assert hasattr(model_class, 'evaluate')
    
    @pytest.mark.skipif(PYTORCH_AVAILABLE, reason="PyTorch is available")
    def test_pytorch_models_import_without_pytorch(self):
        """Test that PyTorch models are None when PyTorch is not available."""
        assert LSTMModel is None
        assert BiLSTMModel is None
        assert GNNModel is None
        assert CNNModel is None


class TestGetClassificationModel:
    """Test the get_classification_model factory function."""
    
    def test_get_random_forest_model(self):
        """Test getting Random Forest model."""
        model = get_classification_model('random_forest', n_estimators=100)
        assert isinstance(model, RandomForestModel)
        assert model.config['n_estimators'] == 100
    
    def test_get_mlp_model(self):
        """Test getting MLP model."""
        model = get_classification_model('mlp', hidden_layer_sizes=(100, 50))
        assert isinstance(model, MLPModel)
        assert model.config['hidden_layer_sizes'] == (100, 50)
    
    def test_get_random_forest_model_case_insensitive(self):
        """Test that model names are case insensitive."""
        model1 = get_classification_model('random_forest')
        model2 = get_classification_model('RANDOM_FOREST')
        model3 = get_classification_model('Random_Forest')
        
        assert isinstance(model1, RandomForestModel)
        assert isinstance(model2, RandomForestModel)
        assert isinstance(model3, RandomForestModel)
    
    def test_get_mlp_model_case_insensitive(self):
        """Test that MLP model names are case insensitive."""
        model1 = get_classification_model('mlp')
        model2 = get_classification_model('MLP')
        model3 = get_classification_model('Mlp')
        
        assert isinstance(model1, MLPModel)
        assert isinstance(model2, MLPModel)
        assert isinstance(model3, MLPModel)
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_get_lstm_model_with_pytorch(self):
        """Test getting LSTM model when PyTorch is available."""
        model = get_classification_model('lstm', input_size=10, hidden_size=64)
        assert isinstance(model, LSTMModel)
        assert model.config['input_size'] == 10
        assert model.config['hidden_size'] == 64
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_get_bilstm_model_with_pytorch(self):
        """Test getting BiLSTM model when PyTorch is available."""
        model = get_classification_model('bilstm', input_size=10, hidden_size=64)
        assert isinstance(model, BiLSTMModel)
        assert model.config['input_size'] == 10
        assert model.config['hidden_size'] == 64
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_get_gnn_model_with_pytorch(self):
        """Test getting GNN model when PyTorch is available."""
        model = get_classification_model('gnn', input_dim=10, hidden_dim=32)
        assert isinstance(model, GNNModel)
        assert model.config['input_dim'] == 10
        assert model.config['hidden_dim'] == 32
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_get_cnn_model_with_pytorch(self):
        """Test getting CNN model when PyTorch is available."""
        model = get_classification_model('cnn', input_channels=10, num_classes=2)
        assert isinstance(model, CNNModel)
        assert model.config['input_channels'] == 10
        assert model.config['num_classes'] == 2
    
    @pytest.mark.skipif(PYTORCH_AVAILABLE, reason="PyTorch is available")
    def test_get_lstm_model_without_pytorch(self):
        """Test getting LSTM model when PyTorch is not available."""
        with pytest.raises(ImportError, match="LSTM model requires PyTorch"):
            get_classification_model('lstm')
    
    @pytest.mark.skipif(PYTORCH_AVAILABLE, reason="PyTorch is available")
    def test_get_bilstm_model_without_pytorch(self):
        """Test getting BiLSTM model when PyTorch is not available."""
        with pytest.raises(ImportError, match="BiLSTM model requires PyTorch"):
            get_classification_model('bilstm')
    
    @pytest.mark.skipif(PYTORCH_AVAILABLE, reason="PyTorch is available")
    def test_get_gnn_model_without_pytorch(self):
        """Test getting GNN model when PyTorch is not available."""
        with pytest.raises(ImportError, match="GNN model requires PyTorch"):
            get_classification_model('gnn')
    
    @pytest.mark.skipif(PYTORCH_AVAILABLE, reason="PyTorch is available")
    def test_get_cnn_model_without_pytorch(self):
        """Test getting CNN model when PyTorch is not available."""
        with pytest.raises(ImportError, match="CNN model requires PyTorch"):
            get_classification_model('cnn')
    
    def test_get_unknown_model(self):
        """Test getting unknown model type."""
        with pytest.raises(ValueError, match="Unknown model name"):
            get_classification_model('unknown_model')
    
    def test_get_unknown_model_with_pytorch_available(self):
        """Test getting unknown model type when PyTorch is available."""
        with pytest.raises(ValueError, match="Unknown model name"):
            get_classification_model('unknown_model')
    
    def test_get_unknown_model_with_pytorch_unavailable(self):
        """Test getting unknown model type when PyTorch is not available."""
        with pytest.raises(ValueError, match="Unknown model name"):
            get_classification_model('unknown_model')
    
    def test_available_models_list_with_pytorch(self):
        """Test that available models list includes PyTorch models when available."""
        if PYTORCH_AVAILABLE:
            # When PyTorch is available, all models should be listed
            with pytest.raises(ValueError) as exc_info:
                get_classification_model('unknown_model')
            error_message = str(exc_info.value)
            assert 'random_forest' in error_message
            assert 'mlp' in error_message
            assert 'lstm' in error_message
            assert 'bilstm' in error_message
            assert 'gnn' in error_message
            assert 'cnn' in error_message
    
    def test_available_models_list_without_pytorch(self):
        """Test that available models list excludes PyTorch models when unavailable."""
        if not PYTORCH_AVAILABLE:
            # When PyTorch is not available, only non-PyTorch models should be listed
            with pytest.raises(ValueError) as exc_info:
                get_classification_model('unknown_model')
            error_message = str(exc_info.value)
            assert 'random_forest' in error_message
            assert 'mlp' in error_message
            assert 'lstm' not in error_message
            assert 'bilstm' not in error_message
            assert 'gnn' not in error_message
            assert 'cnn' not in error_message


class TestModelParameters:
    """Test model parameter passing."""
    
    def test_random_forest_parameters(self):
        """Test passing parameters to Random Forest model."""
        model = get_classification_model(
            'random_forest',
            n_estimators=200,
            max_depth=10,
            random_state=42
        )
        assert model.config['n_estimators'] == 200
        assert model.config['max_depth'] == 10
        assert model.config['random_state'] == 42
    
    def test_mlp_parameters(self):
        """Test passing parameters to MLP model."""
        model = get_classification_model(
            'mlp',
            hidden_layer_sizes=(200, 100, 50),
            activation='tanh',
            solver='lbfgs',
            max_iter=500
        )
        assert model.config['hidden_layer_sizes'] == (200, 100, 50)
        assert model.config['activation'] == 'tanh'
        assert model.config['solver'] == 'lbfgs'
        assert model.config['max_iter'] == 500
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_lstm_parameters(self):
        """Test passing parameters to LSTM model."""
        model = get_classification_model(
            'lstm',
            input_size=20,
            hidden_size=128,
            num_layers=2,
            num_classes=3,
            lr=0.01,
            epochs=50
        )
        assert model.config['input_size'] == 20
        assert model.config['hidden_size'] == 128
        assert model.config['num_layers'] == 2
        assert model.config['num_classes'] == 3
        assert model.config['lr'] == 0.01
        assert model.config['epochs'] == 50
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_cnn_parameters(self):
        """Test passing parameters to CNN model."""
        model = get_classification_model(
            'cnn',
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
    def test_gnn_parameters(self):
        """Test passing parameters to GNN model."""
        model = get_classification_model(
            'gnn',
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


class TestModelFactoryEdgeCases:
    """Test edge cases for the model factory function."""
    
    def test_empty_model_name(self):
        """Test with empty model name."""
        with pytest.raises(ValueError, match="Unknown model name"):
            get_classification_model('')
    
    def test_none_model_name(self):
        """Test with None model name."""
        with pytest.raises(AttributeError):
            get_classification_model(None)
    
    def test_whitespace_model_name(self):
        """Test with whitespace-only model name."""
        with pytest.raises(ValueError, match="Unknown model name"):
            get_classification_model('   ')
    
    def test_numeric_model_name(self):
        """Test with numeric model name."""
        with pytest.raises(ValueError, match="Unknown model name"):
            get_classification_model('123')
    
    def test_special_characters_model_name(self):
        """Test with special characters in model name."""
        with pytest.raises(ValueError, match="Unknown model name"):
            get_classification_model('model@#$%')
    
    def test_partial_model_name(self):
        """Test with partial model name."""
        with pytest.raises(ValueError, match="Unknown model name"):
            get_classification_model('random')  # Should be 'random_forest'
    
    def test_extra_whitespace_model_name(self):
        """Test with extra whitespace in model name."""
        model = get_classification_model('  random_forest  ')
        assert isinstance(model, RandomForestModel)


class TestModelFactoryDocumentation:
    """Test that the factory function follows its documented behavior."""
    
    def test_factory_function_signature(self):
        """Test that the factory function has the expected signature."""
        import inspect
        sig = inspect.signature(get_classification_model)
        
        # Should have 'name' parameter and **kwargs
        assert 'name' in sig.parameters
        assert sig.parameters['name'].annotation == str
        assert 'kwargs' in sig.parameters or any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
    
    def test_factory_function_docstring(self):
        """Test that the factory function has proper documentation."""
        doc = get_classification_model.__doc__
        assert doc is not None
        assert len(doc.strip()) > 0
        assert 'name' in doc
        assert 'kwargs' in doc
        assert 'Returns' in doc
        assert 'Raises' in doc
        assert 'Example' in doc or 'Examples' in doc
    
    def test_factory_function_example_works(self):
        """Test that the documented example works."""
        # The docstring should contain an example that actually works
        # This is a basic test to ensure the example is functional
        try:
            model = get_classification_model('random_forest')
            assert isinstance(model, RandomForestModel)
        except Exception as e:
            pytest.fail(f"Documented example failed: {e}")


class TestModelConsistency:
    """Test consistency between different ways of creating models."""
    
    def test_direct_vs_factory_random_forest(self):
        """Test that direct instantiation and factory function produce equivalent models."""
        # Direct instantiation
        direct_model = RandomForestModel(n_estimators=100, max_depth=5)
        
        # Factory function
        factory_model = get_classification_model('random_forest', n_estimators=100, max_depth=5)
        
        # Both should have the same configuration
        assert direct_model.config['n_estimators'] == factory_model.config['n_estimators']
        assert direct_model.config['max_depth'] == factory_model.config['max_depth']
        assert direct_model.name == factory_model.name
        assert direct_model.description == factory_model.description
    
    def test_direct_vs_factory_mlp(self):
        """Test that direct instantiation and factory function produce equivalent MLP models."""
        # Direct instantiation
        direct_model = MLPModel(hidden_layer_sizes=(100, 50), activation='tanh')
        
        # Factory function
        factory_model = get_classification_model('mlp', hidden_layer_sizes=(100, 50), activation='tanh')
        
        # Both should have the same configuration
        assert direct_model.config['hidden_layer_sizes'] == factory_model.config['hidden_layer_sizes']
        assert direct_model.config['activation'] == factory_model.config['activation']
        assert direct_model.name == factory_model.name
        assert direct_model.description == factory_model.description
