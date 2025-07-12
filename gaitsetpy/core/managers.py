"""
Singleton managers for GaitSetPy components.

This module provides singleton managers for each component type that handle
plugin registration, discovery, and instantiation.

Maintainer: @aharshit123456
"""

import threading
from typing import Dict, List, Type, Any, Optional, Union
from .base_classes import (
    BaseDatasetLoader, 
    BaseFeatureExtractor, 
    BasePreprocessor, 
    BaseEDAAnalyzer, 
    BaseClassificationModel
)


class SingletonMeta(type):
    """
    Metaclass for implementing singleton pattern with thread safety.
    """
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
            return cls._instances[cls]


class BaseManager(metaclass=SingletonMeta):
    """
    Base manager class for all component managers.
    """
    
    def __init__(self):
        self._registry: Dict[str, Type] = {}
        self._instances: Dict[str, Any] = {}
        self._lock = threading.Lock()
    
    def register(self, name: str, component_class: Type):
        """
        Register a component class.
        
        Args:
            name: Name to register the component under
            component_class: Component class to register
        """
        with self._lock:
            self._registry[name] = component_class
    
    def unregister(self, name: str):
        """
        Unregister a component.
        
        Args:
            name: Name of the component to unregister
        """
        with self._lock:
            if name in self._registry:
                del self._registry[name]
            if name in self._instances:
                del self._instances[name]
    
    def get_available_components(self) -> List[str]:
        """
        Get list of available component names.
        
        Returns:
            List of registered component names
        """
        return list(self._registry.keys())
    
    def get_component_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a registered component.
        
        Args:
            name: Name of the component
            
        Returns:
            Component information dictionary or None if not found
        """
        if name in self._registry:
            component_class = self._registry[name]
            return {
                'name': name,
                'class': component_class.__name__,
                'module': component_class.__module__,
                'doc': component_class.__doc__
            }
        return None
    
    def create_instance(self, name: str, *args, **kwargs) -> Any:
        """
        Create an instance of a registered component.
        
        Args:
            name: Name of the component
            *args: Positional arguments for component constructor
            **kwargs: Keyword arguments for component constructor
            
        Returns:
            Component instance
            
        Raises:
            ValueError: If component is not registered
        """
        if name not in self._registry:
            raise ValueError(f"Component '{name}' is not registered")
        
        component_class = self._registry[name]
        return component_class(*args, **kwargs)
    
    def get_cached_instance(self, name: str, *args, **kwargs) -> Any:
        """
        Get a cached instance of a component, creating it if it doesn't exist.
        
        Args:
            name: Name of the component
            *args: Positional arguments for component constructor
            **kwargs: Keyword arguments for component constructor
            
        Returns:
            Component instance
        """
        with self._lock:
            if name not in self._instances:
                self._instances[name] = self.create_instance(name, *args, **kwargs)
            return self._instances[name]


class DatasetManager(BaseManager):
    """
    Singleton manager for dataset loaders.
    """
    
    def register_dataset(self, name: str, dataset_class: Type[BaseDatasetLoader]):
        """
        Register a dataset loader.
        
        Args:
            name: Name to register the dataset under
            dataset_class: Dataset loader class
        """
        if not issubclass(dataset_class, BaseDatasetLoader):
            raise ValueError(f"Dataset class must inherit from BaseDatasetLoader")
        self.register(name, dataset_class)
    
    def load_dataset(self, name: str, data_dir: str, **kwargs) -> BaseDatasetLoader:
        """
        Load a dataset using the registered loader.
        
        Args:
            name: Name of the dataset loader
            data_dir: Directory containing the dataset
            **kwargs: Additional arguments for the loader
            
        Returns:
            Dataset loader instance with loaded data
        """
        loader = self.create_instance(name, name, f"{name} dataset loader")
        loader.load_data(data_dir, **kwargs)
        return loader


class FeatureManager(BaseManager):
    """
    Singleton manager for feature extractors.
    """
    
    def register_extractor(self, name: str, extractor_class: Type[BaseFeatureExtractor]):
        """
        Register a feature extractor.
        
        Args:
            name: Name to register the extractor under
            extractor_class: Feature extractor class
        """
        if not issubclass(extractor_class, BaseFeatureExtractor):
            raise ValueError(f"Extractor class must inherit from BaseFeatureExtractor")
        self.register(name, extractor_class)
    
    def extract_features(self, extractor_name: str, windows: List[Dict], fs: int, **kwargs) -> List[Dict]:
        """
        Extract features using the specified extractor.
        
        Args:
            extractor_name: Name of the feature extractor
            windows: List of sliding window dictionaries
            fs: Sampling frequency
            **kwargs: Additional arguments for feature extraction
            
        Returns:
            List of feature dictionaries
        """
        extractor = self.get_cached_instance(extractor_name, extractor_name, f"{extractor_name} feature extractor")
        return extractor.extract_features(windows, fs, **kwargs)


class PreprocessingManager(BaseManager):
    """
    Singleton manager for preprocessors.
    """
    
    def register_preprocessor(self, name: str, preprocessor_class: Type[BasePreprocessor]):
        """
        Register a preprocessor.
        
        Args:
            name: Name to register the preprocessor under
            preprocessor_class: Preprocessor class
        """
        if not issubclass(preprocessor_class, BasePreprocessor):
            raise ValueError(f"Preprocessor class must inherit from BasePreprocessor")
        self.register(name, preprocessor_class)
    
    def preprocess_data(self, preprocessor_name: str, data: Any, **kwargs) -> Any:
        """
        Preprocess data using the specified preprocessor.
        
        Args:
            preprocessor_name: Name of the preprocessor
            data: Input data to preprocess
            **kwargs: Additional arguments for preprocessing
            
        Returns:
            Preprocessed data
        """
        preprocessor = self.get_cached_instance(preprocessor_name, preprocessor_name, f"{preprocessor_name} preprocessor")
        return preprocessor.fit_transform(data, **kwargs)


class EDAManager(BaseManager):
    """
    Singleton manager for EDA analyzers.
    """
    
    def register_analyzer(self, name: str, analyzer_class: Type[BaseEDAAnalyzer]):
        """
        Register an EDA analyzer.
        
        Args:
            name: Name to register the analyzer under
            analyzer_class: EDA analyzer class
        """
        if not issubclass(analyzer_class, BaseEDAAnalyzer):
            raise ValueError(f"Analyzer class must inherit from BaseEDAAnalyzer")
        self.register(name, analyzer_class)
    
    def analyze_data(self, analyzer_name: str, data: Any, **kwargs) -> Dict[str, Any]:
        """
        Analyze data using the specified analyzer.
        
        Args:
            analyzer_name: Name of the EDA analyzer
            data: Input data to analyze
            **kwargs: Additional arguments for analysis
            
        Returns:
            Analysis results dictionary
        """
        analyzer = self.get_cached_instance(analyzer_name, analyzer_name, f"{analyzer_name} analyzer")
        return analyzer.analyze(data, **kwargs)
    
    def visualize_data(self, analyzer_name: str, data: Any, **kwargs):
        """
        Create visualizations using the specified analyzer.
        
        Args:
            analyzer_name: Name of the EDA analyzer
            data: Input data to visualize
            **kwargs: Additional arguments for visualization
        """
        analyzer = self.get_cached_instance(analyzer_name, analyzer_name, f"{analyzer_name} analyzer")
        analyzer.visualize(data, **kwargs)


class ClassificationManager(BaseManager):
    """
    Singleton manager for classification models.
    """
    
    def register_model(self, name: str, model_class: Type[BaseClassificationModel]):
        """
        Register a classification model.
        
        Args:
            name: Name to register the model under
            model_class: Classification model class
        """
        if not issubclass(model_class, BaseClassificationModel):
            raise ValueError(f"Model class must inherit from BaseClassificationModel")
        self.register(name, model_class)
    
    def train_model(self, model_name: str, features: List[Dict], **kwargs) -> BaseClassificationModel:
        """
        Train a classification model.
        
        Args:
            model_name: Name of the classification model
            features: List of feature dictionaries
            **kwargs: Additional arguments for training
            
        Returns:
            Trained model instance
        """
        model = self.create_instance(model_name, model_name, f"{model_name} classification model")
        model.train(features, **kwargs)
        return model
    
    def predict(self, model_name: str, features: List[Dict], **kwargs) -> Any:
        """
        Make predictions using a trained model.
        
        Args:
            model_name: Name of the classification model
            features: List of feature dictionaries
            **kwargs: Additional arguments for prediction
            
        Returns:
            Predictions array
        """
        model = self.get_cached_instance(model_name, model_name, f"{model_name} classification model")
        return model.predict(features, **kwargs)
    
    def evaluate_model(self, model_name: str, features: List[Dict], **kwargs) -> Dict[str, float]:
        """
        Evaluate a classification model.
        
        Args:
            model_name: Name of the classification model
            features: List of feature dictionaries
            **kwargs: Additional arguments for evaluation
            
        Returns:
            Evaluation metrics dictionary
        """
        model = self.get_cached_instance(model_name, model_name, f"{model_name} classification model")
        return model.evaluate(features, **kwargs) 