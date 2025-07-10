"""
GaitSetPy: A Python Library for Gait Analysis

This library provides tools for loading, preprocessing, and analyzing gait data.
"""

from .dataset import (
    BaseDataset,
    DaphnetDataset,
    MobiFallDataset,
    ArduousDataset
)

from .preprocessing import PreprocessingPipeline

from .features import (
    FeatureExtractor,
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
    calculate_zero_crossing_rate,
    calculate_energy
)

from .classification.models import (
    BaseModel,
    RandomForestModel,
    LSTMModel,
    BiLSTMModel,
    GNNModel,
    MLPModel
)

from .eda.visualization import (
    plot_sensor_data,
    plot_features,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_learning_curve
)

from .main import process_gait_data

__version__ = '0.1.0'

__all__ = [
    # Dataset classes
    'BaseDataset',
    'DaphnetDataset',
    'MobiFallDataset',
    'ArduousDataset',
    
    # Preprocessing
    'PreprocessingPipeline',
    
    # Feature extraction
    'FeatureExtractor',
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
    'calculate_zero_crossing_rate',
    'calculate_energy',
    
    # Models
    'BaseModel',
    'RandomForestModel',
    'LSTMModel',
    'BiLSTMModel',
    'GNNModel',
    'MLPModel',
    
    # Visualization
    'plot_sensor_data',
    'plot_features',
    'plot_confusion_matrix',
    'plot_feature_importance',
    'plot_learning_curve',
    
    # Main functionality
    'process_gait_data'
]
