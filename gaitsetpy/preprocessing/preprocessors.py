'''
Preprocessor classes for gait data preprocessing.

This module contains individual preprocessor classes that inherit from BasePreprocessor
and provide specific preprocessing functionality.

Maintainer: @aharshit123456
'''

from typing import Union, Dict, Any
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from ..core.base_classes import BasePreprocessor


class ClippingPreprocessor(BasePreprocessor):
    """
    Preprocessor for clipping values to a specified range.
    """
    
    def __init__(self, min_val: float = -1, max_val: float = 1):
        super().__init__(
            name="clipping",
            description="Clips values in the data to be within a specified range"
        )
        self.config = {
            'min_val': min_val,
            'max_val': max_val
        }
    
    def fit(self, data: Union[pd.DataFrame, np.ndarray], **kwargs):
        """
        Fit the preprocessor (no fitting needed for clipping).
        
        Args:
            data: Input data to fit on
            **kwargs: Additional arguments
        """
        # Update config with any passed arguments
        self.config.update({k: v for k, v in kwargs.items() if k in ['min_val', 'max_val']})
        self.fitted = True
    
    def transform(self, data: Union[pd.DataFrame, np.ndarray], **kwargs) -> Union[pd.DataFrame, np.ndarray]:
        """
        Clip values in the data to be within the specified range.
        
        Args:
            data: Input data to transform
            **kwargs: Additional arguments
            
        Returns:
            Clipped data
        """
        min_val = kwargs.get('min_val', self.config['min_val'])
        max_val = kwargs.get('max_val', self.config['max_val'])
        
        return np.clip(data, min_val, max_val)


class NoiseRemovalPreprocessor(BasePreprocessor):
    """
    Preprocessor for removing noise using moving average filter.
    """
    
    def __init__(self, window_size: int = 5):
        super().__init__(
            name="noise_removal",
            description="Applies a moving average filter to reduce noise"
        )
        self.config = {
            'window_size': window_size
        }
    
    def fit(self, data: Union[pd.DataFrame, np.ndarray], **kwargs):
        """
        Fit the preprocessor (no fitting needed for noise removal).
        
        Args:
            data: Input data to fit on
            **kwargs: Additional arguments
        """
        self.config.update({k: v for k, v in kwargs.items() if k in ['window_size']})
        self.fitted = True
    
    def transform(self, data: Union[pd.DataFrame, np.ndarray], **kwargs) -> Union[pd.DataFrame, np.ndarray]:
        """
        Apply a moving average filter to reduce noise.
        
        Args:
            data: Input data to transform
            **kwargs: Additional arguments
            
        Returns:
            Noise-reduced data
        """
        window_size = kwargs.get('window_size', self.config['window_size'])
        
        if isinstance(data, pd.DataFrame):
            return data.rolling(window=window_size, center=True).mean().fillna(method="bfill").fillna(method="ffill")
        elif isinstance(data, pd.Series):
            return data.rolling(window=window_size, center=True).mean().fillna(method="bfill").fillna(method="ffill")
        else:
            # For numpy arrays, use uniform filter
            from scipy.ndimage import uniform_filter1d
            return uniform_filter1d(data, size=window_size, mode='nearest')


class OutlierRemovalPreprocessor(BasePreprocessor):
    """
    Preprocessor for removing outliers using Z-score method.
    """
    
    def __init__(self, threshold: float = 3):
        super().__init__(
            name="outlier_removal",
            description="Removes outliers beyond a given threshold using the Z-score method"
        )
        self.config = {
            'threshold': threshold
        }
        self.mean_ = None
        self.std_ = None
    
    def fit(self, data: Union[pd.DataFrame, np.ndarray], **kwargs):
        """
        Fit the preprocessor by computing mean and standard deviation.
        
        Args:
            data: Input data to fit on
            **kwargs: Additional arguments
        """
        self.config.update({k: v for k, v in kwargs.items() if k in ['threshold']})
        
        if isinstance(data, (pd.DataFrame, pd.Series)):
            self.mean_ = data.mean()
            self.std_ = data.std()
        else:
            self.mean_ = np.mean(data)
            self.std_ = np.std(data)
        
        self.fitted = True
    
    def transform(self, data: Union[pd.DataFrame, np.ndarray], **kwargs) -> Union[pd.DataFrame, np.ndarray]:
        """
        Remove outliers beyond the threshold using Z-score method.
        
        Args:
            data: Input data to transform
            **kwargs: Additional arguments
            
        Returns:
            Data with outliers removed
        """
        threshold = kwargs.get('threshold', self.config['threshold'])
        
        if isinstance(data, (pd.DataFrame, pd.Series)):
            z_scores = (data - self.mean_).abs() / self.std_
            return data[z_scores <= threshold]
        else:
            z_scores = np.abs(data - self.mean_) / self.std_
            return data[z_scores <= threshold]


class BaselineRemovalPreprocessor(BasePreprocessor):
    """
    Preprocessor for removing baseline by subtracting the mean.
    """
    
    def __init__(self):
        super().__init__(
            name="baseline_removal",
            description="Removes baseline by subtracting the mean"
        )
        self.mean_ = None
    
    def fit(self, data: Union[pd.DataFrame, np.ndarray], **kwargs):
        """
        Fit the preprocessor by computing the mean.
        
        Args:
            data: Input data to fit on
            **kwargs: Additional arguments
        """
        if isinstance(data, (pd.DataFrame, pd.Series)):
            self.mean_ = data.mean()
        else:
            self.mean_ = np.mean(data)
        
        self.fitted = True
    
    def transform(self, data: Union[pd.DataFrame, np.ndarray], **kwargs) -> Union[pd.DataFrame, np.ndarray]:
        """
        Remove baseline by subtracting the mean.
        
        Args:
            data: Input data to transform
            **kwargs: Additional arguments
            
        Returns:
            Baseline-corrected data
        """
        return data - self.mean_


class DriftRemovalPreprocessor(BasePreprocessor):
    """
    Preprocessor for removing low-frequency drift using high-pass filter.
    """
    
    def __init__(self, cutoff: float = 0.01, fs: int = 100):
        super().__init__(
            name="drift_removal",
            description="Removes low-frequency drift using a high-pass filter"
        )
        self.config = {
            'cutoff': cutoff,
            'fs': fs
        }
    
    def fit(self, data: Union[pd.DataFrame, np.ndarray], **kwargs):
        """
        Fit the preprocessor (no fitting needed for drift removal).
        
        Args:
            data: Input data to fit on
            **kwargs: Additional arguments
        """
        self.config.update({k: v for k, v in kwargs.items() if k in ['cutoff', 'fs']})
        self.fitted = True
    
    def transform(self, data: Union[pd.DataFrame, np.ndarray], **kwargs) -> Union[pd.DataFrame, np.ndarray]:
        """
        Remove low-frequency drift using a high-pass filter.
        
        Args:
            data: Input data to transform
            **kwargs: Additional arguments
            
        Returns:
            Drift-corrected data
        """
        cutoff = kwargs.get('cutoff', self.config['cutoff'])
        fs = kwargs.get('fs', self.config['fs'])
        
        b, a = butter(1, cutoff / (fs / 2), btype='highpass')
        
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return pd.Series(filtfilt(b, a, data), index=data.index)
        else:
            return filtfilt(b, a, data)


class HighFrequencyNoiseRemovalPreprocessor(BasePreprocessor):
    """
    Preprocessor for removing high-frequency noise using low-pass filter.
    """
    
    def __init__(self, cutoff: float = 10, fs: int = 100):
        super().__init__(
            name="high_frequency_noise_removal",
            description="Applies a low-pass filter to remove high-frequency noise"
        )
        self.config = {
            'cutoff': cutoff,
            'fs': fs
        }
    
    def fit(self, data: Union[pd.DataFrame, np.ndarray], **kwargs):
        """
        Fit the preprocessor (no fitting needed for filtering).
        
        Args:
            data: Input data to fit on
            **kwargs: Additional arguments
        """
        self.config.update({k: v for k, v in kwargs.items() if k in ['cutoff', 'fs']})
        self.fitted = True
    
    def transform(self, data: Union[pd.DataFrame, np.ndarray], **kwargs) -> Union[pd.DataFrame, np.ndarray]:
        """
        Apply a low-pass filter to remove high-frequency noise.
        
        Args:
            data: Input data to transform
            **kwargs: Additional arguments
            
        Returns:
            Filtered data
        """
        cutoff = kwargs.get('cutoff', self.config['cutoff'])
        fs = kwargs.get('fs', self.config['fs'])
        
        b, a = butter(1, cutoff / (fs / 2), btype='lowpass')
        
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return pd.Series(filtfilt(b, a, data), index=data.index)
        else:
            return filtfilt(b, a, data)


class LowFrequencyNoiseRemovalPreprocessor(BasePreprocessor):
    """
    Preprocessor for removing low-frequency noise using high-pass filter.
    """
    
    def __init__(self, cutoff: float = 0.5, fs: int = 100):
        super().__init__(
            name="low_frequency_noise_removal",
            description="Applies a high-pass filter to remove low-frequency noise"
        )
        self.config = {
            'cutoff': cutoff,
            'fs': fs
        }
    
    def fit(self, data: Union[pd.DataFrame, np.ndarray], **kwargs):
        """
        Fit the preprocessor (no fitting needed for filtering).
        
        Args:
            data: Input data to fit on
            **kwargs: Additional arguments
        """
        self.config.update({k: v for k, v in kwargs.items() if k in ['cutoff', 'fs']})
        self.fitted = True
    
    def transform(self, data: Union[pd.DataFrame, np.ndarray], **kwargs) -> Union[pd.DataFrame, np.ndarray]:
        """
        Apply a high-pass filter to remove low-frequency noise.
        
        Args:
            data: Input data to transform
            **kwargs: Additional arguments
            
        Returns:
            Filtered data
        """
        cutoff = kwargs.get('cutoff', self.config['cutoff'])
        fs = kwargs.get('fs', self.config['fs'])
        
        b, a = butter(1, cutoff / (fs / 2), btype='highpass')
        
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return pd.Series(filtfilt(b, a, data), index=data.index)
        else:
            return filtfilt(b, a, data)


class ArtifactRemovalPreprocessor(BasePreprocessor):
    """
    Preprocessor for removing artifacts by interpolating missing values.
    """
    
    def __init__(self, method: str = "linear"):
        super().__init__(
            name="artifact_removal",
            description="Removes artifacts by interpolating missing values"
        )
        self.config = {
            'method': method
        }
    
    def fit(self, data: Union[pd.DataFrame, np.ndarray], **kwargs):
        """
        Fit the preprocessor (no fitting needed for interpolation).
        
        Args:
            data: Input data to fit on
            **kwargs: Additional arguments
        """
        self.config.update({k: v for k, v in kwargs.items() if k in ['method']})
        self.fitted = True
    
    def transform(self, data: Union[pd.DataFrame, np.ndarray], **kwargs) -> Union[pd.DataFrame, np.ndarray]:
        """
        Remove artifacts by interpolating missing values.
        
        Args:
            data: Input data to transform
            **kwargs: Additional arguments
            
        Returns:
            Artifact-free data
        """
        method = kwargs.get('method', self.config['method'])
        
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.interpolate(method=method).fillna(method="bfill").fillna(method="ffill")
        else:
            # For numpy arrays, use linear interpolation
            from scipy.interpolate import interp1d
            x = np.arange(len(data))
            valid_mask = ~np.isnan(data)
            if np.any(valid_mask):
                f = interp1d(x[valid_mask], data[valid_mask], kind='linear', fill_value='extrapolate')
                return f(x)
            else:
                return data


class TrendRemovalPreprocessor(BasePreprocessor):
    """
    Preprocessor for removing trends using polynomial fitting.
    """
    
    def __init__(self, order: int = 2):
        super().__init__(
            name="trend_removal",
            description="Removes trends using polynomial fitting"
        )
        self.config = {
            'order': order
        }
    
    def fit(self, data: Union[pd.DataFrame, np.ndarray], **kwargs):
        """
        Fit the preprocessor (no fitting needed for detrending).
        
        Args:
            data: Input data to fit on
            **kwargs: Additional arguments
        """
        self.config.update({k: v for k, v in kwargs.items() if k in ['order']})
        self.fitted = True
    
    def transform(self, data: Union[pd.DataFrame, np.ndarray], **kwargs) -> Union[pd.DataFrame, np.ndarray]:
        """
        Remove trends using polynomial fitting.
        
        Args:
            data: Input data to transform
            **kwargs: Additional arguments
            
        Returns:
            Detrended data
        """
        order = kwargs.get('order', self.config['order'])
        
        if isinstance(data, (pd.DataFrame, pd.Series)):
            x = np.arange(len(data))
            poly_coeffs = np.polyfit(x, data, order)
            trend = np.polyval(poly_coeffs, x)
            return data - trend
        else:
            x = np.arange(len(data))
            poly_coeffs = np.polyfit(x, data, order)
            trend = np.polyval(poly_coeffs, x)
            return data - trend


class DCOffsetRemovalPreprocessor(BasePreprocessor):
    """
    Preprocessor for removing DC offset by subtracting the mean.
    """
    
    def __init__(self):
        super().__init__(
            name="dc_offset_removal",
            description="Removes DC offset by subtracting the mean"
        )
        self.mean_ = None
    
    def fit(self, data: Union[pd.DataFrame, np.ndarray], **kwargs):
        """
        Fit the preprocessor by computing the mean.
        
        Args:
            data: Input data to fit on
            **kwargs: Additional arguments
        """
        if isinstance(data, (pd.DataFrame, pd.Series)):
            self.mean_ = data.mean()
        else:
            self.mean_ = np.mean(data)
        
        self.fitted = True
    
    def transform(self, data: Union[pd.DataFrame, np.ndarray], **kwargs) -> Union[pd.DataFrame, np.ndarray]:
        """
        Remove DC offset by subtracting the mean.
        
        Args:
            data: Input data to transform
            **kwargs: Additional arguments
            
        Returns:
            DC-corrected data
        """
        return data - self.mean_ 