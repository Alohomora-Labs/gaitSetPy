"""
Unit tests for preprocessing functionality in GaitSetPy.

This module tests the preprocessing classes and utilities
including noise removal, outlier detection, and signal processing.

Maintainer: @aharshit123456
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, Mock

from gaitsetpy.preprocessing.preprocessors import (
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


class TestClippingPreprocessor:
    """Test cases for ClippingPreprocessor."""
    
    def test_instantiation(self):
        """Test ClippingPreprocessor instantiation."""
        preprocessor = ClippingPreprocessor()
        assert preprocessor.name == "clipping"
        assert preprocessor.config['min_val'] == -1
        assert preprocessor.config['max_val'] == 1
    
    def test_instantiation_custom_bounds(self):
        """Test ClippingPreprocessor with custom bounds."""
        preprocessor = ClippingPreprocessor(min_val=-5, max_val=5)
        assert preprocessor.config['min_val'] == -5
        assert preprocessor.config['max_val'] == 5
    
    def test_fit(self):
        """Test fit method."""
        preprocessor = ClippingPreprocessor()
        data = np.array([1, 2, 3, 4, 5])
        
        preprocessor.fit(data)
        assert preprocessor.fitted is True
    
    def test_transform_numpy_array(self):
        """Test transform method with numpy array."""
        preprocessor = ClippingPreprocessor(min_val=0, max_val=3)
        data = np.array([-1, 0, 1, 2, 3, 4, 5])
        
        result = preprocessor.transform(data)
        expected = np.array([0, 0, 1, 2, 3, 3, 3])
        
        assert np.array_equal(result, expected)
    
    def test_transform_pandas_dataframe(self):
        """Test transform method with pandas DataFrame."""
        preprocessor = ClippingPreprocessor(min_val=0, max_val=3)
        data = pd.DataFrame({'col1': [-1, 0, 1, 2, 3, 4, 5]})
        
        result = preprocessor.transform(data)
        expected = pd.DataFrame({'col1': [0, 0, 1, 2, 3, 3, 3]})
        
        pd.testing.assert_frame_equal(result, expected)
    
    def test_fit_transform(self):
        """Test fit_transform method."""
        preprocessor = ClippingPreprocessor(min_val=0, max_val=3)
        data = np.array([-1, 0, 1, 2, 3, 4, 5])
        
        result = preprocessor.fit_transform(data)
        expected = np.array([0, 0, 1, 2, 3, 3, 3])
        
        assert np.array_equal(result, expected)
        assert preprocessor.fitted is True


class TestNoiseRemovalPreprocessor:
    """Test cases for NoiseRemovalPreprocessor."""
    
    def test_instantiation(self):
        """Test NoiseRemovalPreprocessor instantiation."""
        preprocessor = NoiseRemovalPreprocessor()
        assert preprocessor.name == "noise_removal"
        assert preprocessor.config['window_size'] == 5
    
    def test_instantiation_custom_window(self):
        """Test NoiseRemovalPreprocessor with custom window size."""
        preprocessor = NoiseRemovalPreprocessor(window_size=10)
        assert preprocessor.config['window_size'] == 10
    
    def test_fit(self):
        """Test fit method."""
        preprocessor = NoiseRemovalPreprocessor()
        data = np.array([1, 2, 3, 4, 5])
        
        preprocessor.fit(data)
        assert preprocessor.fitted is True
    
    def test_transform_numpy_array(self):
        """Test transform method with numpy array."""
        preprocessor = NoiseRemovalPreprocessor(window_size=3)
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        
        result = preprocessor.transform(data)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(data)
    
    def test_transform_pandas_dataframe(self):
        """Test transform method with pandas DataFrame."""
        preprocessor = NoiseRemovalPreprocessor(window_size=3)
        data = pd.DataFrame({'col1': [1, 2, 3, 4, 5, 6, 7, 8, 9]})
        
        result = preprocessor.transform(data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(data)


class TestOutlierRemovalPreprocessor:
    """Test cases for OutlierRemovalPreprocessor."""
    
    def test_instantiation(self):
        """Test OutlierRemovalPreprocessor instantiation."""
        preprocessor = OutlierRemovalPreprocessor()
        assert preprocessor.name == "outlier_removal"
        assert preprocessor.config['threshold'] == 3
    
    def test_instantiation_custom_threshold(self):
        """Test OutlierRemovalPreprocessor with custom threshold."""
        preprocessor = OutlierRemovalPreprocessor(threshold=2)
        assert preprocessor.config['threshold'] == 2
    
    def test_fit_numpy_array(self):
        """Test fit method with numpy array."""
        preprocessor = OutlierRemovalPreprocessor()
        data = np.array([1, 2, 3, 4, 5, 100])  # 100 is an outlier
        
        preprocessor.fit(data)
        assert preprocessor.fitted is True
        assert preprocessor.mean_ is not None
        assert preprocessor.std_ is not None
    
    def test_fit_pandas_dataframe(self):
        """Test fit method with pandas DataFrame."""
        preprocessor = OutlierRemovalPreprocessor()
        data = pd.DataFrame({'col1': [1, 2, 3, 4, 5, 100]})
        
        preprocessor.fit(data)
        assert preprocessor.fitted is True
        assert preprocessor.mean_ is not None
        assert preprocessor.std_ is not None
    
    def test_transform_numpy_array(self):
        """Test transform method with numpy array."""
        preprocessor = OutlierRemovalPreprocessor(threshold=2)
        data = np.array([1, 2, 3, 4, 5, 100])  # 100 is an outlier
        
        preprocessor.fit(data)
        result = preprocessor.transform(data)
        
        # Should remove the outlier
        assert len(result) < len(data)
        assert 100 not in result
    
    def test_transform_pandas_dataframe(self):
        """Test transform method with pandas DataFrame."""
        preprocessor = OutlierRemovalPreprocessor(threshold=2)
        data = pd.DataFrame({'col1': [1, 2, 3, 4, 5, 100]})
        
        preprocessor.fit(data)
        result = preprocessor.transform(data)
        
        # Should replace the outlier with NaN
        assert len(result) == len(data)
        assert np.isnan(result['col1'].iloc[-1])  # Last value should be NaN (outlier)


class TestBaselineRemovalPreprocessor:
    """Test cases for BaselineRemovalPreprocessor."""
    
    def test_instantiation(self):
        """Test BaselineRemovalPreprocessor instantiation."""
        preprocessor = BaselineRemovalPreprocessor()
        assert preprocessor.name == "baseline_removal"
    
    def test_fit_numpy_array(self):
        """Test fit method with numpy array."""
        preprocessor = BaselineRemovalPreprocessor()
        data = np.array([1, 2, 3, 4, 5])
        
        preprocessor.fit(data)
        assert preprocessor.fitted is True
        assert preprocessor.mean_ == 3.0
    
    def test_fit_pandas_dataframe(self):
        """Test fit method with pandas DataFrame."""
        preprocessor = BaselineRemovalPreprocessor()
        data = pd.DataFrame({'col1': [1, 2, 3, 4, 5]})
        
        preprocessor.fit(data)
        assert preprocessor.fitted is True
        assert preprocessor.mean_['col1'] == 3.0
    
    def test_transform_numpy_array(self):
        """Test transform method with numpy array."""
        preprocessor = BaselineRemovalPreprocessor()
        data = np.array([1, 2, 3, 4, 5])
        
        preprocessor.fit(data)
        result = preprocessor.transform(data)
        
        expected = data - 3.0  # Subtract mean
        assert np.array_equal(result, expected)
    
    def test_transform_pandas_dataframe(self):
        """Test transform method with pandas DataFrame."""
        preprocessor = BaselineRemovalPreprocessor()
        data = pd.DataFrame({'col1': [1, 2, 3, 4, 5]})
        
        preprocessor.fit(data)
        result = preprocessor.transform(data)
        
        expected = data - 3.0  # Subtract mean
        pd.testing.assert_frame_equal(result, expected)


class TestDriftRemovalPreprocessor:
    """Test cases for DriftRemovalPreprocessor."""
    
    def test_instantiation(self):
        """Test DriftRemovalPreprocessor instantiation."""
        preprocessor = DriftRemovalPreprocessor()
        assert preprocessor.name == "drift_removal"
        assert preprocessor.config['cutoff'] == 0.01
        assert preprocessor.config['fs'] == 100
    
    def test_instantiation_custom_params(self):
        """Test DriftRemovalPreprocessor with custom parameters."""
        preprocessor = DriftRemovalPreprocessor(cutoff=0.05, fs=200)
        assert preprocessor.config['cutoff'] == 0.05
        assert preprocessor.config['fs'] == 200
    
    def test_fit(self):
        """Test fit method."""
        preprocessor = DriftRemovalPreprocessor()
        data = np.array([1, 2, 3, 4, 5])
        
        preprocessor.fit(data)
        assert preprocessor.fitted is True
    
    def test_transform_numpy_array(self):
        """Test transform method with numpy array."""
        preprocessor = DriftRemovalPreprocessor(cutoff=0.1, fs=100)
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        result = preprocessor.transform(data)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(data)
    
    def test_transform_pandas_dataframe(self):
        """Test transform method with pandas DataFrame."""
        preprocessor = DriftRemovalPreprocessor(cutoff=0.1, fs=100)
        data = pd.DataFrame({'col1': np.random.randn(100)})  # Use longer data for filter
        
        result = preprocessor.transform(data)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(data)


class TestHighFrequencyNoiseRemovalPreprocessor:
    """Test cases for HighFrequencyNoiseRemovalPreprocessor."""
    
    def test_instantiation(self):
        """Test HighFrequencyNoiseRemovalPreprocessor instantiation."""
        preprocessor = HighFrequencyNoiseRemovalPreprocessor()
        assert preprocessor.name == "high_frequency_noise_removal"
        assert preprocessor.config['cutoff'] == 10
        assert preprocessor.config['fs'] == 100
    
    def test_instantiation_custom_params(self):
        """Test HighFrequencyNoiseRemovalPreprocessor with custom parameters."""
        preprocessor = HighFrequencyNoiseRemovalPreprocessor(cutoff=20, fs=200)
        assert preprocessor.config['cutoff'] == 20
        assert preprocessor.config['fs'] == 200
    
    def test_fit(self):
        """Test fit method."""
        preprocessor = HighFrequencyNoiseRemovalPreprocessor()
        data = np.array([1, 2, 3, 4, 5])
        
        preprocessor.fit(data)
        assert preprocessor.fitted is True
    
    def test_transform_numpy_array(self):
        """Test transform method with numpy array."""
        preprocessor = HighFrequencyNoiseRemovalPreprocessor(cutoff=5, fs=100)
        data = np.random.randn(100)
        
        result = preprocessor.transform(data)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(data)
    
    def test_transform_pandas_dataframe(self):
        """Test transform method with pandas DataFrame."""
        preprocessor = HighFrequencyNoiseRemovalPreprocessor(cutoff=5, fs=100)
        data = pd.DataFrame({'col1': np.random.randn(200)})  # Use longer data for filter
        
        result = preprocessor.transform(data)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(data)


class TestLowFrequencyNoiseRemovalPreprocessor:
    """Test cases for LowFrequencyNoiseRemovalPreprocessor."""
    
    def test_instantiation(self):
        """Test LowFrequencyNoiseRemovalPreprocessor instantiation."""
        preprocessor = LowFrequencyNoiseRemovalPreprocessor()
        assert preprocessor.name == "low_frequency_noise_removal"
        assert preprocessor.config['cutoff'] == 0.5
        assert preprocessor.config['fs'] == 100
    
    def test_instantiation_custom_params(self):
        """Test LowFrequencyNoiseRemovalPreprocessor with custom parameters."""
        preprocessor = LowFrequencyNoiseRemovalPreprocessor(cutoff=1.0, fs=200)
        assert preprocessor.config['cutoff'] == 1.0
        assert preprocessor.config['fs'] == 200
    
    def test_fit(self):
        """Test fit method."""
        preprocessor = LowFrequencyNoiseRemovalPreprocessor()
        data = np.array([1, 2, 3, 4, 5])
        
        preprocessor.fit(data)
        assert preprocessor.fitted is True
    
    def test_transform_numpy_array(self):
        """Test transform method with numpy array."""
        preprocessor = LowFrequencyNoiseRemovalPreprocessor(cutoff=0.1, fs=100)
        data = np.random.randn(100)
        
        result = preprocessor.transform(data)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(data)
    
    def test_transform_pandas_dataframe(self):
        """Test transform method with pandas DataFrame."""
        preprocessor = LowFrequencyNoiseRemovalPreprocessor(cutoff=0.1, fs=100)
        data = pd.DataFrame({'col1': np.random.randn(200)})  # Use longer data for filter
        
        result = preprocessor.transform(data)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(data)


class TestArtifactRemovalPreprocessor:
    """Test cases for ArtifactRemovalPreprocessor."""
    
    def test_instantiation(self):
        """Test ArtifactRemovalPreprocessor instantiation."""
        preprocessor = ArtifactRemovalPreprocessor()
        assert preprocessor.name == "artifact_removal"
        assert preprocessor.config['method'] == "linear"
    
    def test_instantiation_custom_method(self):
        """Test ArtifactRemovalPreprocessor with custom method."""
        preprocessor = ArtifactRemovalPreprocessor(method="cubic")
        assert preprocessor.config['method'] == "cubic"
    
    def test_fit(self):
        """Test fit method."""
        preprocessor = ArtifactRemovalPreprocessor()
        data = np.array([1, 2, 3, 4, 5])
        
        preprocessor.fit(data)
        assert preprocessor.fitted is True
    
    def test_transform_numpy_array(self):
        """Test transform method with numpy array."""
        preprocessor = ArtifactRemovalPreprocessor()
        data = np.array([1, 2, np.nan, 4, 5])
        
        result = preprocessor.transform(data)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(data)
        assert not np.any(np.isnan(result))
    
    def test_transform_pandas_dataframe(self):
        """Test transform method with pandas DataFrame."""
        preprocessor = ArtifactRemovalPreprocessor()
        data = pd.DataFrame({'col1': [1, 2, np.nan, 4, 5]})
        
        result = preprocessor.transform(data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(data)
        assert not result['col1'].isna().any()


class TestTrendRemovalPreprocessor:
    """Test cases for TrendRemovalPreprocessor."""
    
    def test_instantiation(self):
        """Test TrendRemovalPreprocessor instantiation."""
        preprocessor = TrendRemovalPreprocessor()
        assert preprocessor.name == "trend_removal"
        assert preprocessor.config['order'] == 2
    
    def test_instantiation_custom_order(self):
        """Test TrendRemovalPreprocessor with custom order."""
        preprocessor = TrendRemovalPreprocessor(order=3)
        assert preprocessor.config['order'] == 3
    
    def test_fit(self):
        """Test fit method."""
        preprocessor = TrendRemovalPreprocessor()
        data = np.array([1, 2, 3, 4, 5])
        
        preprocessor.fit(data)
        assert preprocessor.fitted is True
    
    def test_transform_numpy_array(self):
        """Test transform method with numpy array."""
        preprocessor = TrendRemovalPreprocessor(order=1)
        data = np.array([1, 2, 3, 4, 5])  # Linear trend
        
        result = preprocessor.transform(data)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(data)
    
    def test_transform_pandas_dataframe(self):
        """Test transform method with pandas DataFrame."""
        preprocessor = TrendRemovalPreprocessor(order=1)
        data = pd.DataFrame({'col1': [1, 2, 3, 4, 5]})
        
        # Fit first to compute the trend
        preprocessor.fit(data)
        result = preprocessor.transform(data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(data)


class TestDCOffsetRemovalPreprocessor:
    """Test cases for DCOffsetRemovalPreprocessor."""
    
    def test_instantiation(self):
        """Test DCOffsetRemovalPreprocessor instantiation."""
        preprocessor = DCOffsetRemovalPreprocessor()
        assert preprocessor.name == "dc_offset_removal"
    
    def test_fit_numpy_array(self):
        """Test fit method with numpy array."""
        preprocessor = DCOffsetRemovalPreprocessor()
        data = np.array([1, 2, 3, 4, 5])
        
        preprocessor.fit(data)
        assert preprocessor.fitted is True
        assert preprocessor.mean_ == 3.0
    
    def test_fit_pandas_dataframe(self):
        """Test fit method with pandas DataFrame."""
        preprocessor = DCOffsetRemovalPreprocessor()
        data = pd.DataFrame({'col1': [1, 2, 3, 4, 5]})
        
        preprocessor.fit(data)
        assert preprocessor.fitted is True
        assert preprocessor.mean_['col1'] == 3.0
    
    def test_transform_numpy_array(self):
        """Test transform method with numpy array."""
        preprocessor = DCOffsetRemovalPreprocessor()
        data = np.array([1, 2, 3, 4, 5])
        
        preprocessor.fit(data)
        result = preprocessor.transform(data)
        
        expected = data - 3.0  # Subtract mean
        assert np.array_equal(result, expected)
    
    def test_transform_pandas_dataframe(self):
        """Test transform method with pandas DataFrame."""
        preprocessor = DCOffsetRemovalPreprocessor()
        data = pd.DataFrame({'col1': [1, 2, 3, 4, 5]})
        
        preprocessor.fit(data)
        result = preprocessor.transform(data)
        
        expected = data - 3.0  # Subtract mean
        pd.testing.assert_frame_equal(result, expected)


class TestPreprocessorEdgeCases:
    """Test edge cases for preprocessors."""
    
    def test_empty_data(self):
        """Test preprocessors with empty data."""
        preprocessor = ClippingPreprocessor()
        data = np.array([])
        
        with pytest.raises((ValueError, IndexError)):
            preprocessor.fit(data)
    
    def test_single_value_data(self):
        """Test preprocessors with single value data."""
        preprocessor = ClippingPreprocessor()
        data = np.array([5])
        
        preprocessor.fit(data)
        result = preprocessor.transform(data)
        
        assert len(result) == 1
        assert result[0] == 5
    
    def test_constant_data(self):
        """Test preprocessors with constant data."""
        preprocessor = BaselineRemovalPreprocessor()
        data = np.array([5, 5, 5, 5, 5])
        
        preprocessor.fit(data)
        result = preprocessor.transform(data)
        
        # After removing baseline, all values should be 0
        assert np.allclose(result, 0)
    
    def test_nan_values(self):
        """Test preprocessors with NaN values."""
        preprocessor = ArtifactRemovalPreprocessor()
        data = np.array([1, 2, np.nan, 4, 5])
        
        preprocessor.fit(data)
        result = preprocessor.transform(data)
        
        assert not np.any(np.isnan(result))
    
    def test_inf_values(self):
        """Test preprocessors with infinite values."""
        preprocessor = ClippingPreprocessor(min_val=-10, max_val=10)
        data = np.array([1, 2, np.inf, 4, 5])
        
        preprocessor.fit(data)
        result = preprocessor.transform(data)
        
        assert not np.any(np.isinf(result))
        assert np.all(result <= 10)
        assert np.all(result >= -10)


class TestPreprocessorChaining:
    """Test chaining multiple preprocessors."""
    
    def test_preprocessor_chain(self):
        """Test chaining multiple preprocessors."""
        # Create a signal with trend and noise
        t = np.linspace(0, 1, 100)
        signal = 2 * t + 0.1 * np.random.randn(100)  # Linear trend + noise
        
        # Apply multiple preprocessors
        baseline_remover = BaselineRemovalPreprocessor()
        trend_remover = TrendRemovalPreprocessor(order=1)
        noise_remover = NoiseRemovalPreprocessor(window_size=5)
        
        # Fit and transform in sequence
        baseline_remover.fit(signal)
        signal_no_baseline = baseline_remover.transform(signal)
        
        trend_remover.fit(signal_no_baseline)
        signal_no_trend = trend_remover.transform(signal_no_baseline)
        
        noise_remover.fit(signal_no_trend)
        final_signal = noise_remover.transform(signal_no_trend)
        
        assert len(final_signal) == len(signal)
        assert isinstance(final_signal, np.ndarray)
    
    def test_fit_transform_chain(self):
        """Test fit_transform method in a chain."""
        signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # Apply multiple preprocessors using fit_transform
        baseline_remover = BaselineRemovalPreprocessor()
        clipper = ClippingPreprocessor(min_val=0, max_val=8)
        
        signal_no_baseline = baseline_remover.fit_transform(signal)
        final_signal = clipper.fit_transform(signal_no_baseline)
        
        assert len(final_signal) == len(signal)
        assert np.all(final_signal >= 0)
        assert np.all(final_signal <= 8)
