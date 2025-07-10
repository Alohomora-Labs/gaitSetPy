"""
Feature Extraction for GaitSetPy

This module provides a class-based feature extraction system for gait data.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
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

@dataclass
class FeatureConfig:
    """Configuration for a feature extraction method."""
    name: str
    enabled: bool = True
    params: Dict[str, Any] = None

class FeatureExtractor:
    """Feature extractor for gait data."""
    
    def __init__(self):
        """Initialize the feature extractor with default features."""
        self.time_domain_features = [
            FeatureConfig("mean"),
            FeatureConfig("std_dev"),
            FeatureConfig("variance"),
            FeatureConfig("skewness"),
            FeatureConfig("kurtosis"),
            FeatureConfig("rms"),
            FeatureConfig("range"),
            FeatureConfig("median"),
            FeatureConfig("mode"),
            FeatureConfig("mav"),
            FeatureConfig("mad"),
            FeatureConfig("peak_height"),
            FeatureConfig("stride_time"),
            FeatureConfig("step_time"),
            FeatureConfig("cadence")
        ]
        
        self.frequency_domain_features = [
            FeatureConfig("freezing_index"),
            FeatureConfig("dominant_frequency"),
            FeatureConfig("peak_frequency"),
            FeatureConfig("power_spectral_entropy"),
            FeatureConfig("principal_harmonic_frequency")
        ]
        
        self.statistical_features = [
            FeatureConfig("entropy"),
            FeatureConfig("iqr"),
            FeatureConfig("correlation"),
            FeatureConfig("auto_regression_coefficients"),
            FeatureConfig("zero_crossing_rate"),
            FeatureConfig("energy")
        ]
    
    def extract(self, data: pd.DataFrame, fs: int = 100, **kwargs) -> Dict[str, Any]:
        """
        Extract all enabled features from the data.
        
        Args:
            data (pd.DataFrame): Input data
            fs (int): Sampling frequency
            **kwargs: Additional parameters for feature extraction
            
        Returns:
            Dict[str, Any]: Dictionary of extracted features
        """
        features = {}
        
        # Extract time domain features
        if kwargs.get('time_domain', True):
            for feature in self.time_domain_features:
                if feature.enabled:
                    features[feature.name] = self._extract_feature(data, feature, fs=fs, **kwargs)
        
        # Extract frequency domain features
        if kwargs.get('frequency_domain', True):
            for feature in self.frequency_domain_features:
                if feature.enabled:
                    features[feature.name] = self._extract_feature(data, feature, fs=fs, **kwargs)
        
        # Extract statistical features
        if kwargs.get('statistical', True):
            for feature in self.statistical_features:
                if feature.enabled:
                    features[feature.name] = self._extract_feature(data, feature, fs=fs, **kwargs)
        
        return features
    
    def _extract_feature(self, data: pd.DataFrame, feature: FeatureConfig, **kwargs) -> Any:
        """Extract a single feature."""
        method = getattr(self, f"_extract_{feature.name}", None)
        if method is None:
            raise ValueError(f"Unknown feature: {feature.name}")
            
        params = {**(feature.params or {}), **kwargs}
        return method(data, **params)
    
    def _extract_mean(self, data: pd.DataFrame, **kwargs) -> np.ndarray:
        return calculate_mean(data)
    
    def _extract_std_dev(self, data: pd.DataFrame, **kwargs) -> np.ndarray:
        return calculate_standard_deviation(data)
    
    def _extract_variance(self, data: pd.DataFrame, **kwargs) -> np.ndarray:
        return calculate_variance(data)
    
    def _extract_skewness(self, data: pd.DataFrame, **kwargs) -> np.ndarray:
        return calculate_skewness(data)
    
    def _extract_kurtosis(self, data: pd.DataFrame, **kwargs) -> np.ndarray:
        return calculate_kurtosis(data)
    
    def _extract_rms(self, data: pd.DataFrame, **kwargs) -> np.ndarray:
        return calculate_root_mean_square(data)
    
    def _extract_range(self, data: pd.DataFrame, **kwargs) -> np.ndarray:
        return calculate_range(data)
    
    def _extract_median(self, data: pd.DataFrame, **kwargs) -> np.ndarray:
        return calculate_median(data)
    
    def _extract_mode(self, data: pd.DataFrame, **kwargs) -> np.ndarray:
        return calculate_mode(data)
    
    def _extract_mav(self, data: pd.DataFrame, **kwargs) -> np.ndarray:
        return calculate_mean_absolute_value(data)
    
    def _extract_mad(self, data: pd.DataFrame, **kwargs) -> np.ndarray:
        return calculate_median_absolute_deviation(data)
    
    def _extract_peak_height(self, data: pd.DataFrame, **kwargs) -> np.ndarray:
        return calculate_peak_height(data)
    
    def _extract_stride_time(self, data: pd.DataFrame, fs: int = 100, **kwargs) -> np.ndarray:
        return calculate_stride_times(data, fs)
    
    def _extract_step_time(self, data: pd.DataFrame, fs: int = 100, **kwargs) -> np.ndarray:
        return calculate_step_time(data, fs)
    
    def _extract_cadence(self, data: pd.DataFrame, fs: int = 100, **kwargs) -> np.ndarray:
        return calculate_cadence(data, fs)
    
    def _extract_freezing_index(self, data: pd.DataFrame, fs: int = 100, **kwargs) -> np.ndarray:
        return calculate_freezing_index(data, fs)
    
    def _extract_dominant_frequency(self, data: pd.DataFrame, fs: int = 100, **kwargs) -> np.ndarray:
        return calculate_dominant_frequency(data, fs)
    
    def _extract_peak_frequency(self, data: pd.DataFrame, fs: int = 100, **kwargs) -> np.ndarray:
        return calculate_peak_frequency(data, fs)
    
    def _extract_power_spectral_entropy(self, data: pd.DataFrame, fs: int = 100, **kwargs) -> np.ndarray:
        return calculate_power_spectral_entropy(data, fs)
    
    def _extract_principal_harmonic_frequency(self, data: pd.DataFrame, fs: int = 100, **kwargs) -> np.ndarray:
        return calculate_principal_harmonic_frequency(data, fs)
    
    def _extract_entropy(self, data: pd.DataFrame, **kwargs) -> np.ndarray:
        return calculate_entropy(data)
    
    def _extract_iqr(self, data: pd.DataFrame, **kwargs) -> np.ndarray:
        return calculate_interquartile_range(data)
    
    def _extract_correlation(self, data: pd.DataFrame, **kwargs) -> np.ndarray:
        return calculate_correlation(data)
    
    def _extract_auto_regression_coefficients(self, data: pd.DataFrame, **kwargs) -> np.ndarray:
        return calculate_auto_regression_coefficients(data)
    
    def _extract_zero_crossing_rate(self, data: pd.DataFrame, **kwargs) -> np.ndarray:
        return calculate_zero_crossing_rate(data)
    
    def _extract_energy(self, data: pd.DataFrame, **kwargs) -> np.ndarray:
        return calculate_energy(data)
    
    def enable_feature(self, feature_name: str) -> None:
        """Enable a feature."""
        for feature_list in [self.time_domain_features, self.frequency_domain_features, self.statistical_features]:
            for feature in feature_list:
                if feature.name == feature_name:
                    feature.enabled = True
                    return
        raise ValueError(f"Unknown feature: {feature_name}")
    
    def disable_feature(self, feature_name: str) -> None:
        """Disable a feature."""
        for feature_list in [self.time_domain_features, self.frequency_domain_features, self.statistical_features]:
            for feature in feature_list:
                if feature.name == feature_name:
                    feature.enabled = False
                    return
        raise ValueError(f"Unknown feature: {feature_name}")
    
    def configure_feature(self, feature_name: str, params: Dict[str, Any]) -> None:
        """Configure parameters for a feature."""
        for feature_list in [self.time_domain_features, self.frequency_domain_features, self.statistical_features]:
            for feature in feature_list:
                if feature.name == feature_name:
                    feature.params = {**(feature.params or {}), **params}
                    return
        raise ValueError(f"Unknown feature: {feature_name}")
