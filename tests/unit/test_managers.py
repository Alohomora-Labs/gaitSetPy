"""
Unit tests for singleton managers in GaitSetPy.

This module tests the singleton manager classes that handle
plugin registration, discovery, and instantiation.

Maintainer: @aharshit123456
"""

import pytest
import threading
from unittest.mock import Mock, patch
from typing import Type

from gaitsetpy.core.managers import (
    SingletonMeta,
    BaseManager,
    DatasetManager,
    FeatureManager,
    PreprocessingManager,
    EDAManager,
    ClassificationManager
)
from gaitsetpy.core.base_classes import (
    BaseDatasetLoader,
    BaseFeatureExtractor,
    BasePreprocessor,
    BaseEDAAnalyzer,
    BaseClassificationModel
)


class MockDatasetLoader(BaseDatasetLoader):
    """Mock dataset loader for testing."""
    
    def __init__(self, name: str = "mock_dataset", description: str = "Mock dataset for testing"):
        super().__init__(name, description)
    
    def load_data(self, data_dir: str, **kwargs):
        return [], []
    
    def create_sliding_windows(self, data, names, window_size=192, step_size=32):
        return []
    
    def get_supported_formats(self):
        return ['.txt']


class MockFeatureExtractor(BaseFeatureExtractor):
    """Mock feature extractor for testing."""
    
    def __init__(self, name: str = "mock_extractor", description: str = "Mock feature extractor for testing"):
        super().__init__(name, description)
    
    def extract_features(self, windows, fs, **kwargs):
        return []
    
    def get_feature_names(self):
        return ['feature1', 'feature2']


class MockPreprocessor(BasePreprocessor):
    """Mock preprocessor for testing."""
    
    def __init__(self, name: str = "mock_preprocessor", description: str = "Mock preprocessor for testing"):
        super().__init__(name, description)
    
    def fit(self, data, **kwargs):
        self.fitted = True
    
    def transform(self, data, **kwargs):
        return data


class MockEDAAnalyzer(BaseEDAAnalyzer):
    """Mock EDA analyzer for testing."""
    
    def __init__(self, name: str = "mock_analyzer", description: str = "Mock EDA analyzer for testing"):
        super().__init__(name, description)
    
    def analyze(self, data, **kwargs):
        return {'result': 'analysis'}
    
    def visualize(self, data, **kwargs):
        pass


class MockClassificationModel(BaseClassificationModel):
    """Mock classification model for testing."""
    
    def __init__(self, name: str = "mock_model", description: str = "Mock classification model for testing"):
        super().__init__(name, description)
    
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


class TestSingletonMeta:
    """Test cases for SingletonMeta metaclass."""
    
    def test_singleton_behavior(self):
        """Test that singleton pattern works correctly."""
        class TestSingleton(metaclass=SingletonMeta):
            def __init__(self):
                self.value = 42
        
        instance1 = TestSingleton()
        instance2 = TestSingleton()
        
        assert instance1 is instance2
        assert instance1.value == 42
    
    def test_thread_safety(self):
        """Test that singleton is thread-safe."""
        class TestSingleton(metaclass=SingletonMeta):
            def __init__(self):
                self.value = 0
        
        instances = []
        
        def create_instance():
            instances.append(TestSingleton())
        
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=create_instance)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All instances should be the same
        assert all(instance is instances[0] for instance in instances)


class TestBaseManager:
    """Test cases for BaseManager."""
    
    def test_instantiation(self):
        """Test that BaseManager can be instantiated."""
        manager = BaseManager()
        assert manager._registry == {}
        assert manager._instances == {}
    
    def test_register_component(self):
        """Test registering a component."""
        manager = BaseManager()
        component_class = Mock()
        
        manager.register("test_component", component_class)
        assert "test_component" in manager._registry
        assert manager._registry["test_component"] == component_class
    
    def test_unregister_component(self):
        """Test unregistering a component."""
        manager = BaseManager()
        component_class = Mock()
        
        manager.register("test_component", component_class)
        manager._instances["test_component"] = Mock()
        
        manager.unregister("test_component")
        assert "test_component" not in manager._registry
        assert "test_component" not in manager._instances
    
    def test_get_available_components(self):
        """Test getting available components."""
        manager = BaseManager()
        component_class = Mock()
        
        manager.register("component1", component_class)
        manager.register("component2", component_class)
        
        available = manager.get_available_components()
        assert set(available) == {"component1", "component2"}
    
    def test_get_component_info(self):
        """Test getting component information."""
        manager = BaseManager()
        component_class = Mock()
        component_class.__name__ = "TestComponent"
        component_class.__module__ = "test.module"
        component_class.__doc__ = "Test component"
        
        manager.register("test_component", component_class)
        
        info = manager.get_component_info("test_component")
        assert info['name'] == "test_component"
        assert info['class'] == "TestComponent"
        assert info['module'] == "test.module"
        assert info['doc'] == "Test component"
    
    def test_get_component_info_not_found(self):
        """Test getting component info for non-existent component."""
        manager = BaseManager()
        info = manager.get_component_info("non_existent")
        assert info is None
    
    def test_create_instance(self):
        """Test creating an instance of a registered component."""
        manager = BaseManager()
        component_class = Mock()
        component_class.return_value = "instance"
        
        manager.register("test_component", component_class)
        
        instance = manager.create_instance("test_component", "arg1", "arg2", kwarg1="value1")
        
        component_class.assert_called_once_with("arg1", "arg2", kwarg1="value1")
        assert instance == "instance"
    
    def test_create_instance_not_registered(self):
        """Test creating instance of non-registered component."""
        manager = BaseManager()
        
        with pytest.raises(ValueError, match="Component 'non_existent' is not registered"):
            manager.create_instance("non_existent")
    
    def test_get_cached_instance(self):
        """Test getting cached instance."""
        manager = BaseManager()
        component_class = Mock()
        component_class.return_value = "instance"
        
        manager.register("test_component", component_class)
        
        # First call should create instance
        instance1 = manager.get_cached_instance("test_component")
        assert instance1 == "instance"
        assert component_class.call_count == 1
        
        # Second call should return cached instance
        instance2 = manager.get_cached_instance("test_component")
        assert instance2 == "instance"
        assert component_class.call_count == 1  # Should not be called again
        assert instance1 is instance2


class TestDatasetManager:
    """Test cases for DatasetManager."""
    
    def test_register_dataset(self):
        """Test registering a dataset loader."""
        manager = DatasetManager()
        
        manager.register_dataset("test_dataset", MockDatasetLoader)
        assert "test_dataset" in manager._registry
        assert manager._registry["test_dataset"] == MockDatasetLoader
    
    def test_register_invalid_dataset(self):
        """Test registering invalid dataset loader."""
        manager = DatasetManager()
        
        with pytest.raises(ValueError, match="Dataset class must inherit from BaseDatasetLoader"):
            manager.register_dataset("invalid", str)  # str doesn't inherit from BaseDatasetLoader
    
    def test_load_dataset(self):
        """Test loading a dataset."""
        manager = DatasetManager()
        
        with patch.object(MockDatasetLoader, 'load_data') as mock_load:
            mock_load.return_value = ([], [])
            
            manager.register_dataset("test_dataset", MockDatasetLoader)
            loader = manager.load_dataset("test_dataset", "/path/to/data")
            
            assert isinstance(loader, MockDatasetLoader)
            mock_load.assert_called_once_with("/path/to/data")


class TestFeatureManager:
    """Test cases for FeatureManager."""
    
    def test_register_extractor(self):
        """Test registering a feature extractor."""
        manager = FeatureManager()
        
        manager.register_extractor("test_extractor", MockFeatureExtractor)
        assert "test_extractor" in manager._registry
        assert manager._registry["test_extractor"] == MockFeatureExtractor
    
    def test_register_invalid_extractor(self):
        """Test registering invalid feature extractor."""
        manager = FeatureManager()
        
        with pytest.raises(ValueError, match="Extractor class must inherit from BaseFeatureExtractor"):
            manager.register_extractor("invalid", str)
    
    def test_extract_features(self):
        """Test extracting features."""
        manager = FeatureManager()
        
        with patch.object(MockFeatureExtractor, 'extract_features') as mock_extract:
            mock_extract.return_value = []
            
            manager.register_extractor("test_extractor", MockFeatureExtractor)
            result = manager.extract_features("test_extractor", [], 100)
            
            assert result == []
            mock_extract.assert_called_once_with([], 100)


class TestPreprocessingManager:
    """Test cases for PreprocessingManager."""
    
    def test_register_preprocessor(self):
        """Test registering a preprocessor."""
        manager = PreprocessingManager()
        
        manager.register_preprocessor("test_preprocessor", MockPreprocessor)
        assert "test_preprocessor" in manager._registry
        assert manager._registry["test_preprocessor"] == MockPreprocessor
    
    def test_register_invalid_preprocessor(self):
        """Test registering invalid preprocessor."""
        manager = PreprocessingManager()
        
        with pytest.raises(ValueError, match="Preprocessor class must inherit from BasePreprocessor"):
            manager.register_preprocessor("invalid", str)
    
    def test_preprocess_data(self):
        """Test preprocessing data."""
        manager = PreprocessingManager()
        
        with patch.object(MockPreprocessor, 'fit_transform') as mock_fit_transform:
            mock_fit_transform.return_value = "processed_data"
            
            manager.register_preprocessor("test_preprocessor", MockPreprocessor)
            result = manager.preprocess_data("test_preprocessor", "raw_data")
            
            assert result == "processed_data"
            mock_fit_transform.assert_called_once_with("raw_data")


class TestEDAManager:
    """Test cases for EDAManager."""
    
    def test_register_analyzer(self):
        """Test registering an EDA analyzer."""
        manager = EDAManager()
        
        manager.register_analyzer("test_analyzer", MockEDAAnalyzer)
        assert "test_analyzer" in manager._registry
        assert manager._registry["test_analyzer"] == MockEDAAnalyzer
    
    def test_register_invalid_analyzer(self):
        """Test registering invalid EDA analyzer."""
        manager = EDAManager()
        
        with pytest.raises(ValueError, match="Analyzer class must inherit from BaseEDAAnalyzer"):
            manager.register_analyzer("invalid", str)
    
    def test_analyze_data(self):
        """Test analyzing data."""
        manager = EDAManager()
        
        with patch.object(MockEDAAnalyzer, 'analyze') as mock_analyze:
            mock_analyze.return_value = {'result': 'analysis'}
            
            manager.register_analyzer("test_analyzer", MockEDAAnalyzer)
            result = manager.analyze_data("test_analyzer", "test_data")
            
            assert result == {'result': 'analysis'}
            mock_analyze.assert_called_once_with("test_data")
    
    def test_visualize_data(self):
        """Test visualizing data."""
        manager = EDAManager()
        
        with patch.object(MockEDAAnalyzer, 'visualize') as mock_visualize:
            manager.register_analyzer("test_analyzer", MockEDAAnalyzer)
            manager.visualize_data("test_analyzer", "test_data")
            
            mock_visualize.assert_called_once_with("test_data")


class TestClassificationManager:
    """Test cases for ClassificationManager."""
    
    def test_register_model(self):
        """Test registering a classification model."""
        manager = ClassificationManager()
        
        manager.register_model("test_model", MockClassificationModel)
        assert "test_model" in manager._registry
        assert manager._registry["test_model"] == MockClassificationModel
    
    def test_register_invalid_model(self):
        """Test registering invalid classification model."""
        manager = ClassificationManager()
        
        with pytest.raises(ValueError, match="Model class must inherit from BaseClassificationModel"):
            manager.register_model("invalid", str)
    
    def test_train_model(self):
        """Test training a model."""
        manager = ClassificationManager()
        
        with patch.object(MockClassificationModel, 'train') as mock_train:
            manager.register_model("test_model", MockClassificationModel)
            model = manager.train_model("test_model", [])
            
            assert isinstance(model, MockClassificationModel)
            mock_train.assert_called_once_with([])
    
    def test_predict(self):
        """Test making predictions."""
        manager = ClassificationManager()
        
        with patch.object(MockClassificationModel, 'predict') as mock_predict:
            mock_predict.return_value = [1, 2, 3]
            
            manager.register_model("test_model", MockClassificationModel)
            result = manager.predict("test_model", [])
            
            assert result == [1, 2, 3]
            mock_predict.assert_called_once_with([])
    
    def test_evaluate_model(self):
        """Test evaluating a model."""
        manager = ClassificationManager()
        
        with patch.object(MockClassificationModel, 'evaluate') as mock_evaluate:
            mock_evaluate.return_value = {'accuracy': 0.95}
            
            manager.register_model("test_model", MockClassificationModel)
            result = manager.evaluate_model("test_model", [])
            
            assert result == {'accuracy': 0.95}
            mock_evaluate.assert_called_once_with([])
