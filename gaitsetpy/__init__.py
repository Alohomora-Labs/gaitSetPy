"""
GaitSetPy - A Python package for gait analysis and recognition.

This package provides a comprehensive toolkit for gait data analysis with both
a modern class-based architecture and legacy function-based API for backward compatibility.

Features:
- Modular architecture with singleton design pattern
- Plugin-based system for easy extension
- Comprehensive dataset loaders (Daphnet, MobiFall, Arduous, PhysioNet)
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
    PhysioNetLoader,
    HARUPLoader,
    get_dataset_manager,
    get_available_datasets,
    load_dataset
)

from .features import (
    GaitFeatureExtractor,
    LBPFeatureExtractor,
    FourierSeriesFeatureExtractor,
    PhysioNetFeatureExtractor,
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
# Explicitly import all public exports from submodules instead of using wildcard imports
# This improves code clarity and makes it easier to track what's being exported

# Dataset legacy functions
from .dataset import (
    load_daphnet_data,
    create_sliding_windows,
    load_mobifall_data,
    load_arduous_data,
    load_physionet_data,
    create_physionet_windows,
    load_harup_data,
    create_harup_windows,
    extract_harup_features,
    download_dataset,
    extract_dataset,
    sliding_window
)

# Features legacy functions
from .features import (
    calculate_mean,
    calculate_standard_deviation,
    calculate_variance,
    calculate_skewness,
    calculate_kurtosis,
    calculate_root_mean_square,
    calculate_range,
    calculate_median,
    calculate_mode,
    calculate_mean_absolute_value,
    calculate_median_absolute_deviation,
    calculate_peak_height,
    calculate_stride_times,
    calculate_step_time,
    calculate_cadence,
    calculate_freezing_index,
    calculate_dominant_frequency,
    calculate_peak_frequency,
    calculate_power_spectral_entropy,
    calculate_principal_harmonic_frequency,
    calculate_entropy,
    calculate_interquartile_range,
    calculate_correlation,
    calculate_auto_regression_coefficients,
    get_mean_for_windows,
    get_standard_deviation_for_windows,
    get_variance_for_windows
)

# Preprocessing legacy functions
from .preprocessing import (
    clip_sliding_windows,
    remove_noise,
    remove_outliers,
    remove_baseline,
    remove_drift,
    remove_artifacts,
    remove_trend,
    remove_dc_offset,
    remove_high_frequency_noise,
    remove_low_frequency_noise
)

# EDA legacy functions
from .eda import (
    plot_thigh_data,
    plot_shank_data,
    plot_trunk_data,
    plot_all_data,
    plot_all_thigh_data,
    plot_all_shank_data,
    plot_all_trunk_data,
    plot_all_datasets,
    plot_sensor_with_features
)

# Classification legacy functions
from .classification import (
    create_random_forest_model,
    preprocess_features,
    evaluate_model
)

# Import version from single source of truth
from ._version import __version__, get_version, get_version_info, get_release_info
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

def load_and_analyze_physionet(data_dir: str, window_size: int = 600, step_size: int = 100):
    """
    Complete workflow for loading and analyzing PhysioNet VGRF data.
    
    Args:
        data_dir: Directory to store/find the PhysioNet dataset
        window_size: Size of sliding windows for feature extraction (default: 600)
        step_size: Step size for sliding windows (default: 100)
        
    Returns:
        Dictionary containing data, features, and analysis results
    """
    # Load dataset
    loader = PhysioNetLoader()
    data, names = loader.load_data(data_dir)
    
    # Create sliding windows
    windows = loader.create_sliding_windows(data, names, window_size=window_size, step_size=step_size)
    
    # Extract PhysioNet-specific features
    extractor = PhysioNetFeatureExtractor()
    all_features = []
    
    for window_dict in windows:
        if 'windows' in window_dict:
            features = extractor.extract_features(window_dict['windows'], fs=100)
            all_features.append({
                'name': window_dict['name'],
                'features': features,
                'metadata': window_dict.get('metadata', {})
            })
    
    return {
        'data': data,
        'names': names,
        'windows': windows,
        'features': all_features,
        'labels': loader.get_labels(),
        'loader': loader,
        'extractor': extractor
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
    'PhysioNetLoader',
    'GaitFeatureExtractor',
    'LBPFeatureExtractor',
    'FourierSeriesFeatureExtractor',
    'PhysioNetFeatureExtractor',
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
    'load_and_analyze_physionet',
    'train_gait_classifier',
    
    # Legacy dataset functions
    'load_daphnet_data',
    'create_sliding_windows',
    'load_mobifall_data',
    'load_arduous_data',
    'load_physionet_data',
    'create_physionet_windows',
    'load_harup_data',
    'create_harup_windows',
    'extract_harup_features',
    'download_dataset',
    'extract_dataset',
    'sliding_window',
    
    # Legacy feature functions
    'calculate_mean',
    'calculate_standard_deviation',
    'calculate_variance',
    'calculate_skewness',
    'calculate_kurtosis',
    'calculate_root_mean_square',
    'calculate_range',
    'calculate_median',
    'calculate_mode',
    'calculate_mean_absolute_value',
    'calculate_median_absolute_deviation',
    'calculate_peak_height',
    'calculate_stride_times',
    'calculate_step_time',
    'calculate_cadence',
    'calculate_freezing_index',
    'calculate_dominant_frequency',
    'calculate_peak_frequency',
    'calculate_power_spectral_entropy',
    'calculate_principal_harmonic_frequency',
    'calculate_entropy',
    'calculate_interquartile_range',
    'calculate_correlation',
    'calculate_auto_regression_coefficients',
    'get_mean_for_windows',
    'get_standard_deviation_for_windows',
    'get_variance_for_windows',
    
    # Legacy preprocessing functions
    'clip_sliding_windows',
    'remove_noise',
    'remove_outliers',
    'remove_baseline',
    'remove_drift',
    'remove_artifacts',
    'remove_trend',
    'remove_dc_offset',
    'remove_high_frequency_noise',
    'remove_low_frequency_noise',
    
    # Legacy EDA functions
    'plot_thigh_data',
    'plot_shank_data',
    'plot_trunk_data',
    'plot_all_data',
    'plot_all_thigh_data',
    'plot_all_shank_data',
    'plot_all_trunk_data',
    'plot_all_datasets',
    'plot_sensor_with_features',
    
    # Legacy classification functions
    'create_random_forest_model',
    'preprocess_features',
    'evaluate_model',
]
