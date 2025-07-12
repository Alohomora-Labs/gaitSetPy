"""
GaitSetPy - A Python package for gait analysis and recognition.

This package provides a comprehensive toolkit for gait data analysis with both
a modern class-based architecture and legacy function-based API for backward compatibility.

Features:
- Modular architecture with singleton design pattern
- Plugin-based system for easy extension
- Comprehensive dataset loaders (Daphnet, MobiFall, Arduous)
- Feature extraction and preprocessing pipelines
- Machine learning models for classification
- Exploratory data analysis tools
- Backward compatibility with legacy API

Architecture:
- Core: Base classes and singleton managers
- Dataset: Data loading and preprocessing
- Features: Feature extraction and analysis
- Preprocessing: Data cleaning and transformation
- EDA: Exploratory data analysis and visualization
- Classification: Machine learning models and evaluation

Maintainer: @aharshit123456
"""

# Core architecture components
from .core import (
    BaseDatasetLoader,
    BaseFeatureExtractor,
    BasePreprocessor,
    BaseEDAAnalyzer,
    BaseClassificationModel,
    DatasetManager,
    FeatureManager,
    PreprocessingManager,
    EDAManager,
    ClassificationManager
)

# New class-based API
from .dataset import (
    DaphnetLoader,
    MobiFallLoader,
    ArduousLoader,
    get_dataset_manager,
    get_available_datasets,
    load_dataset
)

from .features import (
    GaitFeatureExtractor,
    get_feature_manager,
    get_available_extractors,
    extract_features
)

from .preprocessing import (
    ClippingPreprocessor,
    NoiseRemovalPreprocessor,
    OutlierRemovalPreprocessor,
    BaselineRemovalPreprocessor,
    DriftRemovalPreprocessor,
    HighFrequencyNoiseRemovalPreprocessor,
    LowFrequencyNoiseRemovalPreprocessor,
    ArtifactRemovalPreprocessor,
    TrendRemovalPreprocessor,
    DCOffsetRemovalPreprocessor,
    get_preprocessing_manager,
    get_available_preprocessors,
    preprocess_data,
    create_preprocessing_pipeline
)

from .eda import (
    DaphnetVisualizationAnalyzer,
    SensorStatisticsAnalyzer,
    get_eda_manager,
    get_available_analyzers,
    analyze_data,
    visualize_data,
    plot_daphnet_data,
    analyze_sensor_statistics,
    plot_sensor_features
)

from .classification import (
    RandomForestModel,
    get_classification_manager,
    get_available_models,
    train_model,
    predict,
    evaluate_model_performance,
    create_random_forest,
    train_random_forest
)

# Legacy API for backward compatibility
from .dataset import *
from .features import *
from .preprocessing import *
from .eda import *
from .classification import *

__version__ = "0.2.0"  # Updated version to reflect new architecture
__author__ = "Harshit Agarwal | Alohomora Labs"

# Convenient access to all managers
def get_all_managers():
    """
    Get all singleton managers.
    
    Returns:
        Dictionary containing all manager instances
    """
    return {
        'dataset': DatasetManager(),
        'feature': FeatureManager(),
        'preprocessing': PreprocessingManager(),
        'eda': EDAManager(),
        'classification': ClassificationManager()
    }

# System information
def get_system_info():
    """
    Get information about the available components in the system.
    
    Returns:
        Dictionary containing system information
    """
    return {
        'version': __version__,
        'author': __author__,
        'available_datasets': get_available_datasets(),
        'available_extractors': get_available_extractors(),
        'available_preprocessors': get_available_preprocessors(),
        'available_analyzers': get_available_analyzers(),
        'available_models': get_available_models(),
        'architecture': 'Modular with singleton design pattern'
    }

# Shortcut functions for common workflows
def load_and_analyze_daphnet(data_dir: str, sensor_type: str = 'all', window_size: int = 192):
    """
    Complete workflow for loading and analyzing Daphnet data.
    
    Args:
        data_dir: Directory containing the Daphnet dataset
        sensor_type: Type of sensor to analyze ('all', 'thigh', 'shank', 'trunk')
        window_size: Size of sliding windows for feature extraction
        
    Returns:
        Dictionary containing data, features, and analysis results
    """
    # Load dataset
    loader = DaphnetLoader()
    data, names = loader.load_data(data_dir)
    
    # Create sliding windows
    windows = loader.create_sliding_windows(data, names, window_size=window_size)
    
    # Extract features
    extractor = GaitFeatureExtractor()
    features = extractor.extract_features(windows[0]['windows'], fs=64)
    
    # Analyze data
    analyzer = DaphnetVisualizationAnalyzer()
    analysis = analyzer.analyze(data)
    
    return {
        'data': data,
        'names': names,
        'windows': windows,
        'features': features,
        'analysis': analysis,
        'loader': loader,
        'extractor': extractor,
        'analyzer': analyzer
    }

def train_gait_classifier(features, model_type: str = 'random_forest', **kwargs):
    """
    Train a gait classification model.
    
    Args:
        features: List of feature dictionaries
        model_type: Type of model to train ('random_forest', etc.)
        **kwargs: Additional arguments for model training
        
    Returns:
        Trained model instance
    """
    if model_type == 'random_forest':
        model = RandomForestModel(**kwargs)
        model.train(features, **kwargs)
        return model
    else:
        raise ValueError(f"Model type '{model_type}' not supported")

__all__ = [
    # Core architecture
    'BaseDatasetLoader',
    'BaseFeatureExtractor', 
    'BasePreprocessor',
    'BaseEDAAnalyzer',
    'BaseClassificationModel',
    'DatasetManager',
    'FeatureManager',
    'PreprocessingManager',
    'EDAManager',
    'ClassificationManager',
    
    # New class-based API
    'DaphnetLoader',
    'MobiFallLoader',
    'ArduousLoader',
    'GaitFeatureExtractor',
    'ClippingPreprocessor',
    'NoiseRemovalPreprocessor',
    'OutlierRemovalPreprocessor',
    'BaselineRemovalPreprocessor',
    'DriftRemovalPreprocessor',
    'HighFrequencyNoiseRemovalPreprocessor',
    'LowFrequencyNoiseRemovalPreprocessor',
    'ArtifactRemovalPreprocessor',
    'TrendRemovalPreprocessor',
    'DCOffsetRemovalPreprocessor',
    'DaphnetVisualizationAnalyzer',
    'SensorStatisticsAnalyzer',
    'RandomForestModel',
    
    # Manager access functions
    'get_dataset_manager',
    'get_feature_manager',
    'get_preprocessing_manager',
    'get_eda_manager',
    'get_classification_manager',
    'get_all_managers',
    
    # Utility functions
    'get_available_datasets',
    'get_available_extractors',
    'get_available_preprocessors',
    'get_available_analyzers',
    'get_available_models',
    'get_system_info',
    
    # Workflow functions
    'load_and_analyze_daphnet',
    'train_gait_classifier',
    
    # Legacy API (imported via *)
    # All legacy functions are included through wildcard imports
]
