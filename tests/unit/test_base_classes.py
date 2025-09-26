"""
Unit tests for base classes in GaitSetPy.

This module tests the abstract base classes that define the interfaces
for all components in the GaitSetPy architecture.

Maintainer: @aharshit123456
"""

import pytest
import pandas as pd
import numpy as np
from abc import ABC
from unittest.mock import Mock, patch

from gaitsetpy.core.base_classes import (
    BaseDatasetLoader,
    BaseFeatureExtractor,
    BasePreprocessor,
    BaseEDAAnalyzer,
    BaseClassificationModel
)


# Helper classes for testing abstract base classes
class MockDatasetLoader(BaseDatasetLoader):
    def load_data(self, data_dir: str, **kwargs):
        return [], []
    
    def create_sliding_windows(self, data, names, window_size=192, step_size=32):
        return []
    
    def get_supported_formats(self):
        return ['.txt']


class MockFeatureExtractor(BaseFeatureExtractor):
    def extract_features(self, windows, fs, **kwargs):
        return []
    
    def get_feature_names(self):
        return ['feature1', 'feature2']


class MockPreprocessor(BasePreprocessor):
    def fit(self, data, **kwargs):
        self.fitted = True
    
    def transform(self, data, **kwargs):
        return data


class MockEDAAnalyzer(BaseEDAAnalyzer):
    def analyze(self, data, **kwargs):
        return {'result': 'analysis'}
    
    def visualize(self, data, **kwargs):
        pass


class MockClassificationModel(BaseClassificationModel):
    def train(self, features, **kwargs):
        self.trained = True
    
    def predict(self, features, **kwargs):
        return [1, 2, 3]
    
    def evaluate(self, features, **kwargs):
        return {'accuracy': 0.95}
    
    def save_model(self, filepath):
        pass
    
    def load_model(self, filepath):
        pass


class TestBaseDatasetLoader:
    """Test cases for BaseDatasetLoader."""
    
    def test_instantiation(self):
        """Test that BaseDatasetLoader can be instantiated with required parameters."""
        loader = MockDatasetLoader("test_dataset", "Test dataset description")
        assert loader.name == "test_dataset"
        assert loader.description == "Test dataset description"
        assert loader.data is None
        assert loader.metadata == {}
    
    def test_get_info(self):
        """Test get_info method returns correct information."""
        loader = MockDatasetLoader("test_dataset", "Test dataset description")
        info = loader.get_info()
        
        assert info['name'] == "test_dataset"
        assert info['description'] == "Test dataset description"
        assert info['metadata'] == {}
        assert 'supported_formats' in info
    
    def test_abstract_methods(self):
        """Test that abstract methods work when implemented."""
        loader = MockDatasetLoader("test_dataset", "Test dataset description")
        
        # Test that methods work (they're implemented)
        data, names = loader.load_data("dummy_path")
        assert data == []
        assert names == []
        
        windows = loader.create_sliding_windows([], [])
        assert windows == []
        
        formats = loader.get_supported_formats()
        assert formats == ['.txt']


class TestBaseFeatureExtractor:
    """Test cases for BaseFeatureExtractor."""
    
    def test_instantiation(self):
        """Test that BaseFeatureExtractor can be instantiated."""
        extractor = MockFeatureExtractor("test_extractor", "Test feature extractor")
        assert extractor.name == "test_extractor"
        assert extractor.description == "Test feature extractor"
        assert extractor.config == {}
    
    def test_configure(self):
        """Test configure method updates config."""
        extractor = MockFeatureExtractor("test_extractor", "Test feature extractor")
        config = {'param1': 'value1', 'param2': 42}
        
        extractor.configure(config)
        assert extractor.config == config
    
    def test_configure_partial(self):
        """Test configure method updates config partially."""
        extractor = MockFeatureExtractor("test_extractor", "Test feature extractor")
        extractor.config = {'existing': 'value'}
        
        config = {'param1': 'value1'}
        extractor.configure(config)
        
        assert extractor.config == {'existing': 'value', 'param1': 'value1'}
    
    def test_get_info(self):
        """Test get_info method returns correct information."""
        extractor = MockFeatureExtractor("test_extractor", "Test feature extractor")
        extractor.config = {'param1': 'value1'}
        
        info = extractor.get_info()
        assert info['name'] == "test_extractor"
        assert info['description'] == "Test feature extractor"
        assert info['config'] == {'param1': 'value1'}
        assert 'feature_names' in info
    
    def test_abstract_methods(self):
        """Test that abstract methods work when implemented."""
        extractor = MockFeatureExtractor("test_extractor", "Test feature extractor")
        
        # Test that methods work (they're implemented)
        features = extractor.extract_features([], 100)
        assert features == []
        
        feature_names = extractor.get_feature_names()
        assert feature_names == ['feature1', 'feature2']


class TestBasePreprocessor:
    """Test cases for BasePreprocessor."""
    
    def test_instantiation(self):
        """Test that BasePreprocessor can be instantiated."""
        preprocessor = MockPreprocessor("test_preprocessor", "Test preprocessor")
        assert preprocessor.name == "test_preprocessor"
        assert preprocessor.description == "Test preprocessor"
        assert preprocessor.config == {}
        assert preprocessor.fitted is False
    
    def test_configure(self):
        """Test configure method updates config."""
        preprocessor = MockPreprocessor("test_preprocessor", "Test preprocessor")
        config = {'param1': 'value1', 'param2': 42}
        
        preprocessor.configure(config)
        assert preprocessor.config == config
    
    def test_fit_transform(self):
        """Test fit_transform method calls fit and transform."""
        preprocessor = MockPreprocessor("test_preprocessor", "Test preprocessor")
        
        # Mock the fit and transform methods
        preprocessor.fit = Mock()
        preprocessor.transform = Mock(return_value="transformed_data")
        
        result = preprocessor.fit_transform("test_data")
        
        preprocessor.fit.assert_called_once_with("test_data")
        preprocessor.transform.assert_called_once_with("test_data")
        assert result == "transformed_data"
    
    def test_get_info(self):
        """Test get_info method returns correct information."""
        preprocessor = MockPreprocessor("test_preprocessor", "Test preprocessor")
        preprocessor.config = {'param1': 'value1'}
        preprocessor.fitted = True
        
        info = preprocessor.get_info()
        assert info['name'] == "test_preprocessor"
        assert info['description'] == "Test preprocessor"
        assert info['config'] == {'param1': 'value1'}
        assert info['fitted'] is True
    
    def test_abstract_methods(self):
        """Test that abstract methods work when implemented."""
        preprocessor = MockPreprocessor("test_preprocessor", "Test preprocessor")
        
        # Test that methods work (they're implemented)
        preprocessor.fit("test_data")
        assert preprocessor.fitted is True
        
        result = preprocessor.transform("test_data")
        assert result == "test_data"


class TestBaseEDAAnalyzer:
    """Test cases for BaseEDAAnalyzer."""
    
    def test_instantiation(self):
        """Test that BaseEDAAnalyzer can be instantiated."""
        analyzer = MockEDAAnalyzer("test_analyzer", "Test EDA analyzer")
        assert analyzer.name == "test_analyzer"
        assert analyzer.description == "Test EDA analyzer"
        assert analyzer.config == {}
    
    def test_configure(self):
        """Test configure method updates config."""
        analyzer = MockEDAAnalyzer("test_analyzer", "Test EDA analyzer")
        config = {'param1': 'value1', 'param2': 42}
        
        analyzer.configure(config)
        assert analyzer.config == config
    
    def test_get_info(self):
        """Test get_info method returns correct information."""
        analyzer = MockEDAAnalyzer("test_analyzer", "Test EDA analyzer")
        analyzer.config = {'param1': 'value1'}
        
        info = analyzer.get_info()
        assert info['name'] == "test_analyzer"
        assert info['description'] == "Test EDA analyzer"
        assert info['config'] == {'param1': 'value1'}
    
    def test_abstract_methods(self):
        """Test that abstract methods work when implemented."""
        analyzer = MockEDAAnalyzer("test_analyzer", "Test EDA analyzer")
        
        # Test that methods work (they're implemented)
        result = analyzer.analyze("test_data")
        assert result == {'result': 'analysis'}
        
        # visualize method should not raise an error
        analyzer.visualize("test_data")


class TestBaseClassificationModel:
    """Test cases for BaseClassificationModel."""
    
    def test_instantiation(self):
        """Test that BaseClassificationModel can be instantiated."""
        model = MockClassificationModel("test_model", "Test classification model")
        assert model.name == "test_model"
        assert model.description == "Test classification model"
        assert model.model is None
        assert model.config == {}
        assert model.trained is False
    
    def test_configure(self):
        """Test configure method updates config."""
        model = MockClassificationModel("test_model", "Test classification model")
        config = {'param1': 'value1', 'param2': 42}
        
        model.configure(config)
        assert model.config == config
    
    def test_get_info(self):
        """Test get_info method returns correct information."""
        model = MockClassificationModel("test_model", "Test classification model")
        model.config = {'param1': 'value1'}
        model.trained = True
        
        info = model.get_info()
        assert info['name'] == "test_model"
        assert info['description'] == "Test classification model"
        assert info['config'] == {'param1': 'value1'}
        assert info['trained'] is True
    
    def test_abstract_methods(self):
        """Test that abstract methods work when implemented."""
        model = MockClassificationModel("test_model", "Test classification model")
        
        # Test that methods work (they're implemented)
        model.train([])
        assert model.trained is True
        
        predictions = model.predict([])
        assert predictions == [1, 2, 3]
        
        metrics = model.evaluate([])
        assert metrics == {'accuracy': 0.95}
        
        # save_model and load_model should not raise errors
        model.save_model("dummy_path")
        model.load_model("dummy_path")


class TestBaseClassInheritance:
    """Test that base classes properly implement ABC."""
    
    def test_base_classes_are_abstract(self):
        """Test that all base classes are abstract."""
        assert issubclass(BaseDatasetLoader, ABC)
        assert issubclass(BaseFeatureExtractor, ABC)
        assert issubclass(BasePreprocessor, ABC)
        assert issubclass(BaseEDAAnalyzer, ABC)
        assert issubclass(BaseClassificationModel, ABC)
    
    def test_cannot_instantiate_without_implementation(self):
        """Test that base classes cannot be instantiated without implementing abstract methods."""
        # These should raise errors during instantiation because they're abstract
        with pytest.raises(TypeError):
            BaseDatasetLoader("test", "test")
        
        with pytest.raises(TypeError):
            BaseFeatureExtractor("test", "test")
        
        with pytest.raises(TypeError):
            BasePreprocessor("test", "test")
        
        with pytest.raises(TypeError):
            BaseEDAAnalyzer("test", "test")
        
        with pytest.raises(TypeError):
            BaseClassificationModel("test", "test")
