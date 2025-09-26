"""
Unit tests for feature extraction functionality in GaitSetPy.

This module tests the feature extraction classes and utilities
including time-domain, frequency-domain, and statistical features.

Maintainer: @aharshit123456
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, Mock

from gaitsetpy.features.gait_features import GaitFeatureExtractor
from gaitsetpy.features.utils import (
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
    calculate_energy,
)


class TestStatisticalFeatures:
    """Test cases for statistical feature calculations."""
    
    def test_calculate_mean(self):
        """Test mean calculation."""
        data = np.array([1, 2, 3, 4, 5])
        result = calculate_mean(data)
        assert result == 3.0
    
    def test_calculate_standard_deviation(self):
        """Test standard deviation calculation."""
        data = np.array([1, 2, 3, 4, 5])
        result = calculate_standard_deviation(data)
        expected = np.std(data)
        assert abs(result - expected) < 1e-10
    
    def test_calculate_variance(self):
        """Test variance calculation."""
        data = np.array([1, 2, 3, 4, 5])
        result = calculate_variance(data)
        expected = np.var(data)
        assert abs(result - expected) < 1e-10
    
    def test_calculate_skewness(self):
        """Test skewness calculation."""
        data = np.array([1, 2, 3, 4, 5])
        result = calculate_skewness(data)
        assert isinstance(result, float)
        assert not np.isnan(result)
    
    def test_calculate_kurtosis(self):
        """Test kurtosis calculation."""
        data = np.array([1, 2, 3, 4, 5])
        result = calculate_kurtosis(data)
        assert isinstance(result, float)
        assert not np.isnan(result)
    
    def test_calculate_root_mean_square(self):
        """Test RMS calculation."""
        data = np.array([1, 2, 3, 4, 5])
        result = calculate_root_mean_square(data)
        expected = np.sqrt(np.mean(data**2))
        assert abs(result - expected) < 1e-10
    
    def test_calculate_range(self):
        """Test range calculation."""
        data = np.array([1, 2, 3, 4, 5])
        result = calculate_range(data)
        assert result == 4.0
    
    def test_calculate_median(self):
        """Test median calculation."""
        data = np.array([1, 2, 3, 4, 5])
        result = calculate_median(data)
        assert result == 3.0
    
    def test_calculate_mode(self):
        """Test mode calculation."""
        data = np.array([1, 2, 2, 3, 3, 3])
        result = calculate_mode(data)
        assert result == 3
    
    def test_calculate_mean_absolute_value(self):
        """Test mean absolute value calculation."""
        data = np.array([-1, 2, -3, 4, -5])
        result = calculate_mean_absolute_value(data)
        expected = np.mean(np.abs(data))
        assert abs(result - expected) < 1e-10
    
    def test_calculate_median_absolute_deviation(self):
        """Test median absolute deviation calculation."""
        data = np.array([1, 2, 3, 4, 5])
        result = calculate_median_absolute_deviation(data)
        median = np.median(data)
        expected = np.median(np.abs(data - median))
        assert abs(result - expected) < 1e-10
    
    def test_calculate_peak_height(self):
        """Test peak height calculation."""
        data = np.array([1, 2, 5, 2, 1])
        result = calculate_peak_height(data)
        assert result == 5.0
    
    def test_calculate_entropy(self):
        """Test entropy calculation."""
        data = np.array([1, 2, 3, 4, 5])
        result = calculate_entropy(data)
        assert isinstance(result, float)
        assert not np.isnan(result)
    
    def test_calculate_interquartile_range(self):
        """Test interquartile range calculation."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = calculate_interquartile_range(data)
        q75, q25 = np.percentile(data, [75, 25])
        expected = q75 - q25
        assert abs(result - expected) < 1e-10
    
    def test_calculate_correlation(self):
        """Test correlation calculation."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        result = calculate_correlation(x, y)
        expected = np.corrcoef(x, y)[0, 1]
        assert abs(result - expected) < 1e-10
    
    def test_calculate_zero_crossing_rate(self):
        """Test zero crossing rate calculation."""
        data = np.array([-1, 1, -1, 1, -1])
        result = calculate_zero_crossing_rate(data)
        assert result > 0
    
    def test_calculate_energy(self):
        """Test energy calculation."""
        data = np.array([1, 2, 3, 4, 5])
        result = calculate_energy(data)
        expected = np.sum(data**2)
        assert abs(result - expected) < 1e-10


class TestFrequencyDomainFeatures:
    """Test cases for frequency domain feature calculations."""
    
    def test_calculate_dominant_frequency(self):
        """Test dominant frequency calculation."""
        # Create a sine wave
        fs = 100
        t = np.linspace(0, 1, fs)
        data = np.sin(2 * np.pi * 5 * t)  # 5 Hz sine wave
        result = calculate_dominant_frequency(data, fs)
        assert isinstance(result, float)
        assert not np.isnan(result)
    
    def test_calculate_peak_frequency(self):
        """Test peak frequency calculation."""
        fs = 100
        t = np.linspace(0, 1, fs)
        data = np.sin(2 * np.pi * 5 * t)
        result = calculate_peak_frequency(data, fs)
        assert isinstance(result, float)
        assert not np.isnan(result)
    
    def test_calculate_power_spectral_entropy(self):
        """Test power spectral entropy calculation."""
        fs = 100
        t = np.linspace(0, 1, fs)
        data = np.sin(2 * np.pi * 5 * t)
        result = calculate_power_spectral_entropy(data, fs)
        assert isinstance(result, float)
        assert not np.isnan(result)
    
    def test_calculate_principal_harmonic_frequency(self):
        """Test principal harmonic frequency calculation."""
        fs = 100
        t = np.linspace(0, 1, fs)
        data = np.sin(2 * np.pi * 5 * t)
        result = calculate_principal_harmonic_frequency(data, fs)
        assert isinstance(result, float)
        assert not np.isnan(result)
    
    def test_calculate_stride_times(self):
        """Test stride times calculation."""
        fs = 100
        t = np.linspace(0, 1, fs)
        data = np.sin(2 * np.pi * 5 * t)
        result = calculate_stride_times(data, fs)
        assert isinstance(result, float)
        assert not np.isnan(result)
    
    def test_calculate_step_time(self):
        """Test step time calculation."""
        fs = 100
        t = np.linspace(0, 1, fs)
        data = np.sin(2 * np.pi * 5 * t)
        result = calculate_step_time(data, fs)
        assert isinstance(result, (float, np.ndarray))
        if isinstance(result, np.ndarray):
            assert len(result) > 0
            assert not np.any(np.isnan(result))
        else:
            assert not np.isnan(result)
    
    def test_calculate_cadence(self):
        """Test cadence calculation."""
        fs = 100
        t = np.linspace(0, 1, fs)
        data = np.sin(2 * np.pi * 5 * t)
        result = calculate_cadence(data, fs)
        assert isinstance(result, float)
        assert not np.isnan(result)
    
    def test_calculate_freezing_index(self):
        """Test freezing index calculation."""
        fs = 100
        t = np.linspace(0, 1, fs)
        data = np.sin(2 * np.pi * 5 * t)
        result = calculate_freezing_index(data, fs)
        assert isinstance(result, float)
        assert not np.isnan(result)


class TestAutoRegressionFeatures:
    """Test cases for auto-regression feature calculations."""
    
    def test_calculate_auto_regression_coefficients(self):
        """Test auto-regression coefficients calculation."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = calculate_auto_regression_coefficients(data, order=3)
        assert isinstance(result, (list, np.ndarray))
        assert len(result) >= 3  # May return more coefficients than requested


class TestGaitFeatureExtractor:
    """Test cases for GaitFeatureExtractor class."""
    
    def test_instantiation(self):
        """Test GaitFeatureExtractor instantiation."""
        extractor = GaitFeatureExtractor(verbose=False)
        assert extractor.name == "gait_features"
        assert "gait feature extractor" in extractor.description.lower()
        assert extractor.config['time_domain'] is True
        assert extractor.config['frequency_domain'] is True
        assert extractor.config['statistical'] is True
    
    def test_instantiation_verbose(self):
        """Test GaitFeatureExtractor instantiation with verbose output."""
        with patch('builtins.print') as mock_print:
            extractor = GaitFeatureExtractor(verbose=True)
            mock_print.assert_called()
    
    def test_get_feature_names(self):
        """Test getting feature names."""
        extractor = GaitFeatureExtractor(verbose=False)
        feature_names = extractor.get_feature_names()
        
        assert isinstance(feature_names, list)
        assert 'mean' in feature_names
        assert 'std' in feature_names
        assert 'dominant_frequency' in feature_names
        assert 'entropy' in feature_names
        assert 'ar_coefficients' in feature_names
    
    def test_extract_features_basic(self, sample_sliding_windows):
        """Test basic feature extraction."""
        extractor = GaitFeatureExtractor(verbose=False)
        fs = 64
        
        features = extractor.extract_features(sample_sliding_windows, fs)
        
        assert isinstance(features, list)
        assert len(features) > 0
        
        for feature_dict in features:
            assert 'name' in feature_dict
            assert 'features' in feature_dict
    
    def test_extract_features_time_domain_only(self, sample_sliding_windows):
        """Test feature extraction with only time domain features."""
        extractor = GaitFeatureExtractor(verbose=False)
        fs = 64
        
        features = extractor.extract_features(
            sample_sliding_windows, fs,
            time_domain=True,
            frequency_domain=False,
            statistical=False
        )
        
        assert isinstance(features, list)
        # Check that time domain features are present
        for feature_dict in features:
            if feature_dict['name'] != 'annotations':
                assert 'mean' in feature_dict['features']
                assert 'std' in feature_dict['features']
    
    def test_extract_features_frequency_domain_only(self, sample_sliding_windows):
        """Test feature extraction with only frequency domain features."""
        extractor = GaitFeatureExtractor(verbose=False)
        fs = 64
        
        features = extractor.extract_features(
            sample_sliding_windows, fs,
            time_domain=False,
            frequency_domain=True,
            statistical=False
        )
        
        assert isinstance(features, list)
        # Check that frequency domain features are present
        for feature_dict in features:
            if feature_dict['name'] != 'annotations':
                assert 'dominant_frequency' in feature_dict['features']
    
    def test_extract_features_statistical_only(self, sample_sliding_windows):
        """Test feature extraction with only statistical features."""
        extractor = GaitFeatureExtractor(verbose=False)
        fs = 64
        
        features = extractor.extract_features(
            sample_sliding_windows, fs,
            time_domain=False,
            frequency_domain=False,
            statistical=True
        )
        
        assert isinstance(features, list)
        # Check that statistical features are present
        for feature_dict in features:
            if feature_dict['name'] != 'annotations':
                assert 'entropy' in feature_dict['features']
    
    def test_extract_features_annotations(self, sample_sliding_windows):
        """Test feature extraction with annotations."""
        extractor = GaitFeatureExtractor(verbose=False)
        fs = 64
        
        features = extractor.extract_features(sample_sliding_windows, fs)
        
        # Find annotations feature
        annotation_feature = None
        for feature_dict in features:
            if feature_dict['name'] == 'annotations':
                annotation_feature = feature_dict
                break
        
        assert annotation_feature is not None
        assert 'annotations' in annotation_feature
    
    def test_extract_features_empty_windows(self):
        """Test feature extraction with empty windows."""
        extractor = GaitFeatureExtractor(verbose=False)
        fs = 64
        
        features = extractor.extract_features([], fs)
        assert features == []
    
    def test_extract_features_single_window(self):
        """Test feature extraction with single window."""
        extractor = GaitFeatureExtractor(verbose=False)
        fs = 64
        
        # Create single window
        window = [{
            'name': 'test_sensor',
            'data': [pd.Series([1, 2, 3, 4, 5], name='test_sensor')]
        }]
        
        features = extractor.extract_features(window, fs)
        
        assert isinstance(features, list)
        assert len(features) == 1
        assert features[0]['name'] == 'test_sensor'
    
    def test_print_extraction_summary(self, sample_features):
        """Test printing extraction summary."""
        extractor = GaitFeatureExtractor(verbose=False)
        
        with patch('builtins.print') as mock_print:
            extractor.print_extraction_summary(sample_features)
            mock_print.assert_called()
    
    def test_configure(self):
        """Test configuring the extractor."""
        extractor = GaitFeatureExtractor(verbose=False)
        
        config = {'ar_order': 5, 'new_param': 'value'}
        extractor.configure(config)
        
        assert extractor.config['ar_order'] == 5
        assert extractor.config['new_param'] == 'value'
    
    def test_ensure_numpy_array(self):
        """Test _ensure_numpy_array method."""
        extractor = GaitFeatureExtractor(verbose=False)
        
        # Test with pandas Series
        series = pd.Series([1, 2, 3, 4, 5])
        result = extractor._ensure_numpy_array(series)
        assert isinstance(result, np.ndarray)
        
        # Test with numpy array
        array = np.array([1, 2, 3, 4, 5])
        result = extractor._ensure_numpy_array(array)
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, array)
    
    def test_extract_annotation_labels(self):
        """Test _extract_annotation_labels method."""
        extractor = GaitFeatureExtractor(verbose=False)
        
        # Test with pandas Series
        series = pd.Series([1, 2, 2, 1, 2])
        result = extractor._extract_annotation_labels(series)
        assert result == 2  # Most common value
        
        # Test with numpy array
        array = np.array([1, 2, 2, 1, 2])
        result = extractor._extract_annotation_labels(array)
        assert result == 2


class TestFeatureExtractionEdgeCases:
    """Test edge cases in feature extraction."""
    
    def test_empty_data(self):
        """Test feature extraction with empty data."""
        extractor = GaitFeatureExtractor(verbose=False)
        
        # Test with empty array
        result = calculate_mean(np.array([]))
        assert np.isnan(result)
        
        result = calculate_standard_deviation(np.array([]))
        assert np.isnan(result)
    
    def test_single_value_data(self):
        """Test feature extraction with single value."""
        extractor = GaitFeatureExtractor(verbose=False)
        
        data = np.array([5])
        assert calculate_mean(data) == 5.0
        assert calculate_standard_deviation(data) == 0.0
        assert calculate_median(data) == 5.0
    
    def test_constant_data(self):
        """Test feature extraction with constant data."""
        extractor = GaitFeatureExtractor(verbose=False)
        
        data = np.array([5, 5, 5, 5, 5])
        assert calculate_mean(data) == 5.0
        assert calculate_standard_deviation(data) == 0.0
        assert calculate_variance(data) == 0.0
    
    def test_nan_values(self):
        """Test feature extraction with NaN values."""
        extractor = GaitFeatureExtractor(verbose=False)
        
        data = np.array([1, 2, np.nan, 4, 5])
        
        # Most functions will propagate NaN values
        result = calculate_mean(data)
        assert np.isnan(result)  # NaN propagates through mean calculation
    
    def test_inf_values(self):
        """Test feature extraction with infinite values."""
        extractor = GaitFeatureExtractor(verbose=False)
        
        data = np.array([1, 2, np.inf, 4, 5])
        
        # Most functions will propagate inf values
        result = calculate_mean(data)
        assert np.isinf(result)  # inf propagates through mean calculation


class TestFeatureExtractionPerformance:
    """Test performance aspects of feature extraction."""
    
    def test_large_data_performance(self):
        """Test feature extraction performance with large data."""
        extractor = GaitFeatureExtractor(verbose=False)
        
        # Create large dataset
        large_data = np.random.randn(10000)
        
        import time
        start_time = time.time()
        
        # Test multiple features
        calculate_mean(large_data)
        calculate_standard_deviation(large_data)
        calculate_entropy(large_data)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert execution_time < 1.0  # 1 second threshold
    
    def test_memory_usage(self):
        """Test memory usage during feature extraction."""
        extractor = GaitFeatureExtractor(verbose=False)
        
        # Create moderately large dataset
        data = np.random.randn(1000, 100)  # 1000 samples, 100 features
        
        # Extract features should not cause memory issues
        for i in range(data.shape[1]):
            calculate_mean(data[:, i])
            calculate_standard_deviation(data[:, i])
            calculate_entropy(data[:, i])
