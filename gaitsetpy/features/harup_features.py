'''
HAR-UP Feature Extractor.
Maintainer: @aharshit123456

This file contains the HAR-UP feature extractor class that inherits from BaseFeatureExtractor.
It implements the feature extraction methods used in the HAR-UP project.

Reference:
- Website: https://sites.google.com/up.edu.mx/har-up/
- GitHub: https://github.com/jpnm561/HAR-UP
'''

import numpy as np
from typing import List, Dict, Any
from scipy.stats import kurtosis, skew
from scipy.fftpack import rfft
from ..core.base_classes import BaseFeatureExtractor


class HARUPFeatureExtractor(BaseFeatureExtractor):
    """
    HAR-UP feature extractor class.
    
    This class implements the feature extraction methods used in the HAR-UP project.
    It extracts both time-domain and frequency-domain features from sensor data.
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the HAR-UP feature extractor.
        
        Args:
            verbose: Whether to print progress information
        """
        super().__init__(
            name="harup",
            description="HAR-UP Feature Extractor - Extracts features used in the HAR-UP project"
        )
        self.config = {
            'time_domain': True,
            'frequency_domain': True,
            'verbose': verbose
        }
        
        # Define the features to extract
        self.time_domain_features = [
            'mean', 'std', 'rms', 'max_amp', 'min_amp', 'median',
            'zero_crossings', 'skewness', 'kurtosis', 'q1', 'q3', 'autocorr'
        ]
        
        self.freq_domain_features = [
            'energy'
        ]
    
    def extract_features(self, windows: List[Dict], fs: int, **kwargs) -> List[Dict]:
        """
        Extract features from sliding windows.
        
        Args:
            windows: List of sliding window dictionaries
            fs: Sampling frequency
            **kwargs: Additional arguments for feature extraction
            
        Returns:
            List of feature dictionaries
        """
        # Update config with kwargs
        self.config.update(kwargs)
        
        all_features = []
        
        # Skip label and activity_id windows
        sensor_windows = [w for w in windows if w["name"] not in ["labels", "activity_id"]]
        
        if self.config['verbose']:
            print(f"Extracting features from {len(sensor_windows)} sensor windows")
        
        # Process each sensor window
        for window in sensor_windows:
            sensor_name = window["name"]
            sensor_data = window["data"]
            
            if self.config['verbose']:
                print(f"Processing {sensor_name} with {len(sensor_data)} windows")
            
            # For each window of this sensor
            for i, window_data in enumerate(sensor_data):
                features = {}
                features["sensor"] = sensor_name
                
                # Time domain features
                if self.config['time_domain']:
                    self._extract_time_domain_features(window_data, features)
                
                # Frequency domain features
                if self.config['frequency_domain']:
                    self._extract_freq_domain_features(window_data, features)
                
                all_features.append(features)
        
        return all_features
    
    def _extract_time_domain_features(self, window_data: np.ndarray, features: Dict[str, Any]):
        """
        Extract time domain features from a window.
        
        Args:
            window_data: Window data
            features: Dictionary to store the extracted features
        """
        # Basic statistical features
        features["mean"] = np.mean(window_data)
        features["std"] = np.std(window_data)
        features["rms"] = np.sqrt(np.mean(window_data**2))
        features["max_amp"] = np.max(np.abs(window_data))
        features["min_amp"] = np.min(np.abs(window_data))
        features["median"] = np.median(window_data)
        
        # Zero crossings
        zero_crossings = np.where(np.diff(np.signbit(window_data)))[0]
        features["zero_crossings"] = len(zero_crossings)
        
        # Higher-order statistics
        features["skewness"] = skew(window_data)
        features["kurtosis"] = kurtosis(window_data)
        
        # Quartiles
        features["q1"] = np.percentile(window_data, 25)
        features["q3"] = np.percentile(window_data, 75)
        
        # Autocorrelation
        autocorr = np.correlate(window_data, window_data, mode='full')
        features["autocorr"] = np.median(autocorr)
    
    def _extract_freq_domain_features(self, window_data: np.ndarray, features: Dict[str, Any]):
        """
        Extract frequency domain features from a window.
        
        Args:
            window_data: Window data
            features: Dictionary to store the extracted features
        """
        # FFT
        fft_values = abs(rfft(np.asarray(window_data)))
        
        # Energy
        features["energy"] = np.sum(fft_values**2)
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of features extracted by this extractor.
        
        Returns:
            List of feature names
        """
        feature_names = []
        
        if self.config['time_domain']:
            feature_names.extend(self.time_domain_features)
        
        if self.config['frequency_domain']:
            feature_names.extend(self.freq_domain_features)
        
        return feature_names


# Legacy function wrapper for backward compatibility
def extract_harup_features(windows: List[Dict], fs: int = 100, 
                         time_domain: bool = True, freq_domain: bool = True, 
                         verbose: bool = False) -> List[Dict]:
    """
    Legacy function for extracting HAR-UP features.
    
    Args:
        windows: List of sliding window dictionaries
        fs: Sampling frequency (default: 100Hz)
        time_domain: Whether to extract time domain features
        freq_domain: Whether to extract frequency domain features
        verbose: Whether to print progress information
        
    Returns:
        List of feature dictionaries
    """
    extractor = HARUPFeatureExtractor(verbose=verbose)
    return extractor.extract_features(
        windows, 
        fs=fs, 
        time_domain=time_domain, 
        frequency_domain=freq_domain
    )