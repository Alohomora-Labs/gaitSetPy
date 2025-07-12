"""
preprocessing: Preprocessing pipelines for gait data.

This module provides both the new class-based preprocessors and legacy function-based API.
All preprocessors inherit from BasePreprocessor and are registered with the PreprocessingManager.

Features:
- Clipping and normalization
- Noise removal (moving average, frequency filtering)
- Outlier detection and removal
- Baseline and drift correction
- Artifact removal and trend removal
- DC offset correction

Maintainer: @aharshit123456
"""

# Import the new class-based preprocessors
from .preprocessors import (
    ClippingPreprocessor,
    NoiseRemovalPreprocessor,
    OutlierRemovalPreprocessor,
    BaselineRemovalPreprocessor,
    DriftRemovalPreprocessor,
    HighFrequencyNoiseRemovalPreprocessor,
    LowFrequencyNoiseRemovalPreprocessor,
    ArtifactRemovalPreprocessor,
    TrendRemovalPreprocessor,
    DCOffsetRemovalPreprocessor
)

# Import legacy functions for backward compatibility
from .pipeline import (
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

# Import managers
from ..core.managers import PreprocessingManager

# Register all preprocessors with the manager
def _register_preprocessors():
    """Register all available preprocessors with the PreprocessingManager."""
    manager = PreprocessingManager()
    manager.register_preprocessor("clipping", ClippingPreprocessor)
    manager.register_preprocessor("noise_removal", NoiseRemovalPreprocessor)
    manager.register_preprocessor("outlier_removal", OutlierRemovalPreprocessor)
    manager.register_preprocessor("baseline_removal", BaselineRemovalPreprocessor)
    manager.register_preprocessor("drift_removal", DriftRemovalPreprocessor)
    manager.register_preprocessor("high_frequency_noise_removal", HighFrequencyNoiseRemovalPreprocessor)
    manager.register_preprocessor("low_frequency_noise_removal", LowFrequencyNoiseRemovalPreprocessor)
    manager.register_preprocessor("artifact_removal", ArtifactRemovalPreprocessor)
    manager.register_preprocessor("trend_removal", TrendRemovalPreprocessor)
    manager.register_preprocessor("dc_offset_removal", DCOffsetRemovalPreprocessor)

# Auto-register preprocessors when module is imported
_register_preprocessors()

# Convenient access to the preprocessing manager
def get_preprocessing_manager():
    """Get the singleton PreprocessingManager instance."""
    return PreprocessingManager()

# Helper function to get available preprocessors
def get_available_preprocessors():
    """Get list of available preprocessor names."""
    return PreprocessingManager().get_available_components()

# Helper function to preprocess data using manager
def preprocess_data(preprocessor_name: str, data, **kwargs):
    """
    Preprocess data using the PreprocessingManager.
    
    Args:
        preprocessor_name: Name of the preprocessor
        data: Input data to preprocess
        **kwargs: Additional arguments for preprocessing
        
    Returns:
        Preprocessed data
    """
    return PreprocessingManager().preprocess_data(preprocessor_name, data, **kwargs)

# Pipeline function for chaining multiple preprocessors
def create_preprocessing_pipeline(preprocessor_names: list, **kwargs):
    """
    Create a preprocessing pipeline with multiple preprocessors.
    
    Args:
        preprocessor_names: List of preprocessor names to chain
        **kwargs: Additional arguments for individual preprocessors
        
    Returns:
        Function that applies all preprocessors in sequence
    """
    manager = PreprocessingManager()
    
    def pipeline(data):
        processed_data = data
        for name in preprocessor_names:
            preprocessor = manager.get_cached_instance(name, name, f"{name} preprocessor")
            processed_data = preprocessor.fit_transform(processed_data, **kwargs.get(name, {}))
        return processed_data
    
    return pipeline

__all__ = [
    # New class-based preprocessors
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
    # Legacy functions for backward compatibility
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
    # Manager functions
    'get_preprocessing_manager',
    'get_available_preprocessors',
    'preprocess_data',
    'create_preprocessing_pipeline'
]
