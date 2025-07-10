"""
Features Module for GaitSetPy

This module provides feature extraction functionality for gait analysis.
"""

from .gait_features import FeatureExtractor
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
    calculate_auto_regression_coefficients,
    calculate_zero_crossing_rate,
    calculate_energy
)

__all__ = [
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
    'calculate_energy'
]