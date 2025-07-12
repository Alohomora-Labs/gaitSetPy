'''
Features module for gait analysis and feature extraction.

This module provides both the new class-based feature extractors and legacy function-based API.
All feature extractors inherit from BaseFeatureExtractor and are registered with the FeatureManager.

Maintainer: @aharshit123456
'''

# Import the new class-based feature extractor
from .gait_features import GaitFeatureExtractor

# Import legacy functions for backward compatibility
from .utils import (
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
    calculate_auto_regression_coefficients
)

from .gait_features import (
    get_mean_for_windows,
    get_standard_deviation_for_windows,
    get_variance_for_windows,
    get_skewness_for_windows,
    get_kurtosis_for_windows,
    get_root_mean_square_for_windows,
    get_range_for_windows,
    get_median_for_windows,
    get_mode_for_windows,
    get_mean_absolute_value_for_windows,
    get_median_absolute_deviation_for_windows,
    get_peak_height_for_windows,
    get_stride_times_for_windows,
    get_step_times_for_windows,
    get_cadence_for_windows,
    get_freezing_index_for_windows,
    get_dominant_frequency_for_windows,
    get_peak_frequency_for_windows,
    get_power_spectral_entropy_for_windows,
    get_principal_harmonic_frequency_for_windows,
    get_entropy_for_windows,
    get_interquartile_range_for_windows,
    get_correlation_for_windows,
    get_auto_regression_coefficients_for_windows,
    extract_gait_features
)

# Import managers
from ..core.managers import FeatureManager

# Register all feature extractors with the manager
def _register_extractors():
    """Register all available feature extractors with the FeatureManager."""
    manager = FeatureManager()
    manager.register_extractor("gait_features", GaitFeatureExtractor)

# Auto-register extractors when module is imported
_register_extractors()

# Convenient access to the feature manager
def get_feature_manager():
    """Get the singleton FeatureManager instance."""
    return FeatureManager()

# Helper function to get available extractors
def get_available_extractors():
    """Get list of available feature extractor names."""
    return FeatureManager().get_available_components()

# Helper function to extract features using manager
def extract_features(extractor_name: str, windows, fs: int, **kwargs):
    """
    Extract features using the FeatureManager.
    
    Args:
        extractor_name: Name of the feature extractor
        windows: List of sliding window dictionaries
        fs: Sampling frequency
        **kwargs: Additional arguments for feature extraction
        
    Returns:
        List of feature dictionaries
    """
    return FeatureManager().extract_features(extractor_name, windows, fs, **kwargs)

__all__ = [
    # New class-based extractor
    'GaitFeatureExtractor',
    # Legacy utility functions
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
    # Legacy window functions
    'get_mean_for_windows',
    'get_standard_deviation_for_windows',
    'get_variance_for_windows',
    'get_skewness_for_windows',
    'get_kurtosis_for_windows',
    'get_root_mean_square_for_windows',
    'get_range_for_windows',
    'get_median_for_windows',
    'get_mode_for_windows',
    'get_mean_absolute_value_for_windows',
    'get_median_absolute_deviation_for_windows',
    'get_peak_height_for_windows',
    'get_stride_times_for_windows',
    'get_step_times_for_windows',
    'get_cadence_for_windows',
    'get_freezing_index_for_windows',
    'get_dominant_frequency_for_windows',
    'get_peak_frequency_for_windows',
    'get_power_spectral_entropy_for_windows',
    'get_principal_harmonic_frequency_for_windows',
    'get_entropy_for_windows',
    'get_interquartile_range_for_windows',
    'get_correlation_for_windows',
    'get_auto_regression_coefficients_for_windows',
    'extract_gait_features',
    # Manager functions
    'get_feature_manager',
    'get_available_extractors',
    'extract_features'
]