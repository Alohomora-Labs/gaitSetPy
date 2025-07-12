'''
PhysioNet VGRF Feature Extractor.
Maintainer: @aharshit123456

This module contains feature extractors specific to the PhysioNet VGRF dataset,
including Local Binary Pattern (LBP) and Fourier series analysis.
'''

from typing import List, Dict, Any
import numpy as np
import pandas as pd
from scipy.integrate import simpson
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy import fftpack
import logging
from tqdm import tqdm
from ..core.base_classes import BaseFeatureExtractor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LBPFeatureExtractor(BaseFeatureExtractor):
    """
    Local Binary Pattern (LBP) feature extractor for VGRF data.
    
    This extractor converts time-series data into LBP codes and extracts
    histogram features from the LBP representation.
    """
    
    def __init__(self, verbose: bool = True):
        super().__init__(
            name="lbp_features",
            description="Local Binary Pattern feature extractor for VGRF time-series data"
        )
        self.verbose = verbose
        self.config = {
            'radius': 2,  # LBP radius (number of neighbors)
            'n_bins': 256,  # Number of histogram bins
            'normalize': True  # Normalize histogram
        }
        
        if self.verbose:
            print("🔍 LBP Feature Extractor initialized!")
    
    def lbp_1d(self, data: np.ndarray, radius: int = 2) -> str:
        """
        Compute 1D Local Binary Pattern for time-series data.
        
        Args:
            data: Input time-series data
            radius: Radius for LBP computation
            
        Returns:
            LBP code as binary string
        """
        n = len(data)
        lbp_code = ''
        
        for i in range(n):
            pattern = ''
            for j in range(i - radius, i + radius + 1):
                if j < 0 or j >= n:
                    pattern += '0'
                elif data[j] >= data[i]:
                    pattern += '1'
                else:
                    pattern += '0'
            lbp_code += pattern
        
        return lbp_code
    
    def lbp_to_histogram(self, lbp_code: str, n_bins: int = 256, normalize: bool = True) -> np.ndarray:
        """
        Convert LBP code to histogram features.
        
        Args:
            lbp_code: Binary LBP code string
            n_bins: Number of histogram bins
            normalize: Whether to normalize histogram
            
        Returns:
            Histogram features as numpy array
        """
        # Convert LBP code to integer values
        if len(lbp_code) == 0:
            return np.zeros(n_bins)
        
        # Process LBP code in chunks of 8 bits (or smaller)
        chunk_size = 8
        lbp_values = []
        
        for i in range(0, len(lbp_code), chunk_size):
            chunk = lbp_code[i:i + chunk_size]
            if len(chunk) > 0:
                # Convert binary string to integer
                try:
                    value = int(chunk, 2)
                    lbp_values.append(value % n_bins)  # Ensure within bin range
                except ValueError:
                    continue
        
        if len(lbp_values) == 0:
            return np.zeros(n_bins)
        
        # Create histogram
        hist, _ = np.histogram(lbp_values, bins=n_bins, range=(0, n_bins))
        
        if normalize and np.sum(hist) > 0:
            hist = hist / np.sum(hist)
        
        return hist
    
    def extract_features(self, windows: List[Dict], fs: int, **kwargs) -> List[Dict]:
        """
        Extract LBP features from sliding windows.
        
        Args:
            windows: List of sliding window dictionaries
            fs: Sampling frequency (unused for LBP)
            **kwargs: Additional arguments
            
        Returns:
            List of feature dictionaries
        """
        # Update config with any passed arguments
        radius = kwargs.get('radius', self.config['radius'])
        n_bins = kwargs.get('n_bins', self.config['n_bins'])
        normalize = kwargs.get('normalize', self.config['normalize'])
        
        if self.verbose:
            print(f"\n🔍 LBP Feature Extraction")
            print(f"📊 Radius: {radius}, Bins: {n_bins}, Normalize: {normalize}")
        
        features = []
        
        for window_dict in tqdm(windows, desc="Processing LBP features", disable=not self.verbose):
            sensor_name = window_dict['name']
            window_data = window_dict['data']
            
            # Skip annotation windows
            if sensor_name == 'annotations':
                continue
            
            sensor_features = {'name': sensor_name, 'features': {}}
            
            # Extract LBP features for each window
            lbp_histograms = []
            lbp_means = []
            lbp_stds = []
            
            for window in window_data:
                # Ensure window is numpy array
                if hasattr(window, 'values'):
                    window = window.values
                
                # Compute LBP
                lbp_code = self.lbp_1d(window, radius)
                
                # Convert to histogram
                hist = self.lbp_to_histogram(lbp_code, n_bins, normalize)
                lbp_histograms.append(hist)
                
                # Extract summary statistics
                lbp_means.append(np.mean(hist))
                lbp_stds.append(np.std(hist))
            
            # Store features
            sensor_features['features'] = {
                'lbp_histograms': lbp_histograms,
                'lbp_mean': lbp_means,
                'lbp_std': lbp_stds,
                'lbp_energy': [np.sum(hist**2) for hist in lbp_histograms],
                'lbp_entropy': [self._calculate_entropy(hist) for hist in lbp_histograms]
            }
            
            features.append(sensor_features)
        
        return features
    
    def _calculate_entropy(self, hist: np.ndarray) -> float:
        """Calculate entropy of histogram."""
        # Avoid log(0) by adding small value
        hist = hist + 1e-10
        return -np.sum(hist * np.log2(hist))
    
    def get_feature_names(self) -> List[str]:
        """Get names of LBP features."""
        return [
            'lbp_histograms', 'lbp_mean', 'lbp_std', 
            'lbp_energy', 'lbp_entropy'
        ]


class FourierSeriesFeatureExtractor(BaseFeatureExtractor):
    """
    Fourier Series feature extractor for VGRF data.
    
    This extractor fits Fourier series to time-series data and extracts
    coefficients and reconstruction features.
    """
    
    def __init__(self, verbose: bool = True):
        super().__init__(
            name="fourier_features",
            description="Fourier series feature extractor for VGRF time-series data"
        )
        self.verbose = verbose
        self.config = {
            'n_terms': 10,  # Number of Fourier terms
            'period': 3.0,  # Period for Fourier series
            'extract_coefficients': True,
            'extract_reconstruction_error': True
        }
        
        if self.verbose:
            print("🌊 Fourier Series Feature Extractor initialized!")
    
    def fit_fourier_series(self, signal: np.ndarray, time_points: np.ndarray, 
                          period: float = 3.0, n_terms: int = 10) -> Dict[str, Any]:
        """
        Fit Fourier series to signal.
        
        Args:
            signal: Input signal
            time_points: Time points
            period: Period of the Fourier series
            n_terms: Number of Fourier terms
            
        Returns:
            Dictionary containing Fourier series parameters
        """
        try:
            # Calculate Fourier coefficients
            L = period
            
            # Calculate a0 (DC component)
            a0 = 2/L * simpson(signal, time_points)
            
            # Calculate an and bn coefficients
            an = []
            bn = []
            
            for n in range(1, n_terms + 1):
                # Calculate an coefficient
                an_val = 2.0/L * simpson(signal * np.cos(2.*np.pi*n*time_points/L), time_points)
                an.append(an_val)
                
                # Calculate bn coefficient
                bn_val = 2.0/L * simpson(signal * np.sin(2.*np.pi*n*time_points/L), time_points)
                bn.append(bn_val)
            
            # Reconstruct signal
            reconstructed = np.full_like(time_points, a0/2)
            for n in range(n_terms):
                reconstructed += an[n] * np.cos(2.*np.pi*(n+1)*time_points/L)
                reconstructed += bn[n] * np.sin(2.*np.pi*(n+1)*time_points/L)
            
            # Calculate reconstruction error
            reconstruction_error = np.mean((signal - reconstructed)**2)
            
            return {
                'a0': a0,
                'an': an,
                'bn': bn,
                'reconstructed': reconstructed,
                'reconstruction_error': reconstruction_error,
                'fourier_energy': a0**2 + 2*np.sum(np.array(an)**2 + np.array(bn)**2)
            }
            
        except Exception as e:
            if self.verbose:
                print(f"Error in Fourier series fitting: {e}")
            return {
                'a0': 0,
                'an': [0] * n_terms,
                'bn': [0] * n_terms,
                'reconstructed': np.zeros_like(time_points),
                'reconstruction_error': float('inf'),
                'fourier_energy': 0
            }
    
    def extract_features(self, windows: List[Dict], fs: int, **kwargs) -> List[Dict]:
        """
        Extract Fourier series features from sliding windows.
        
        Args:
            windows: List of sliding window dictionaries
            fs: Sampling frequency
            **kwargs: Additional arguments
            
        Returns:
            List of feature dictionaries
        """
        # Update config with any passed arguments
        n_terms = kwargs.get('n_terms', self.config['n_terms'])
        period = kwargs.get('period', self.config['period'])
        
        if self.verbose:
            print(f"\n🌊 Fourier Series Feature Extraction")
            print(f"📊 Terms: {n_terms}, Period: {period}")
        
        features = []
        
        for window_dict in tqdm(windows, desc="Processing Fourier features", disable=not self.verbose):
            sensor_name = window_dict['name']
            window_data = window_dict['data']
            
            # Skip annotation windows
            if sensor_name == 'annotations':
                continue
            
            sensor_features = {'name': sensor_name, 'features': {}}
            
            # Extract Fourier features for each window
            a0_values = []
            an_values = []
            bn_values = []
            reconstruction_errors = []
            fourier_energies = []
            
            for window in window_data:
                # Ensure window is numpy array
                if hasattr(window, 'values'):
                    window = window.values
                
                # Create time points
                time_points = np.linspace(0, period, len(window))
                
                # Fit Fourier series
                fourier_result = self.fit_fourier_series(window, time_points, period, n_terms)
                
                # Store results
                a0_values.append(fourier_result['a0'])
                an_values.append(fourier_result['an'])
                bn_values.append(fourier_result['bn'])
                reconstruction_errors.append(fourier_result['reconstruction_error'])
                fourier_energies.append(fourier_result['fourier_energy'])
            
            # Store features
            sensor_features['features'] = {
                'fourier_a0': a0_values,
                'fourier_an': an_values,
                'fourier_bn': bn_values,
                'fourier_reconstruction_error': reconstruction_errors,
                'fourier_energy': fourier_energies,
                'fourier_an_mean': [np.mean(an) for an in an_values],
                'fourier_bn_mean': [np.mean(bn) for bn in bn_values],
                'fourier_an_std': [np.std(an) for an in an_values],
                'fourier_bn_std': [np.std(bn) for bn in bn_values]
            }
            
            features.append(sensor_features)
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get names of Fourier series features."""
        return [
            'fourier_a0', 'fourier_an', 'fourier_bn', 
            'fourier_reconstruction_error', 'fourier_energy',
            'fourier_an_mean', 'fourier_bn_mean',
            'fourier_an_std', 'fourier_bn_std'
        ]


class PhysioNetFeatureExtractor(BaseFeatureExtractor):
    """
    Combined feature extractor for PhysioNet VGRF data.
    
    This extractor combines LBP and Fourier series features along with
    basic statistical features specific to VGRF data.
    """
    
    def __init__(self, verbose: bool = True):
        super().__init__(
            name="physionet_features",
            description="Combined feature extractor for PhysioNet VGRF data including LBP and Fourier features"
        )
        self.verbose = verbose
        self.lbp_extractor = LBPFeatureExtractor(verbose=False)
        self.fourier_extractor = FourierSeriesFeatureExtractor(verbose=False)
        
        if self.verbose:
            print("🚀 PhysioNet Feature Extractor initialized!")
    
    def extract_features(self, windows: List[Dict], fs: int, **kwargs) -> List[Dict]:
        """
        Extract combined features from sliding windows.
        
        Args:
            windows: List of sliding window dictionaries
            fs: Sampling frequency
            **kwargs: Additional arguments
            
        Returns:
            List of feature dictionaries
        """
        # Extract features from each extractor
        extract_lbp = kwargs.get('extract_lbp', True)
        extract_fourier = kwargs.get('extract_fourier', True)
        extract_statistical = kwargs.get('extract_statistical', True)
        
        if self.verbose:
            print(f"\n🔍 PhysioNet Feature Extraction")
            print(f"📊 LBP: {extract_lbp}, Fourier: {extract_fourier}, Statistical: {extract_statistical}")
        
        features = []
        
        # Extract LBP features
        if extract_lbp:
            lbp_features = self.lbp_extractor.extract_features(windows, fs, **kwargs)
        else:
            lbp_features = []
        
        # Extract Fourier features
        if extract_fourier:
            fourier_features = self.fourier_extractor.extract_features(windows, fs, **kwargs)
        else:
            fourier_features = []
        
        # Extract statistical features
        if extract_statistical:
            statistical_features = self._extract_statistical_features(windows)
        else:
            statistical_features = []
        
        # Combine features
        for i, window_dict in enumerate(windows):
            sensor_name = window_dict['name']
            
            # Skip annotation windows
            if sensor_name == 'annotations':
                continue
            
            combined_features = {'name': sensor_name, 'features': {}}
            
            # Add LBP features
            if extract_lbp and i < len(lbp_features):
                combined_features['features'].update(lbp_features[i]['features'])
            
            # Add Fourier features
            if extract_fourier and i < len(fourier_features):
                combined_features['features'].update(fourier_features[i]['features'])
            
            # Add statistical features
            if extract_statistical and i < len(statistical_features):
                combined_features['features'].update(statistical_features[i]['features'])
            
            features.append(combined_features)
        
        return features
    
    def _extract_statistical_features(self, windows: List[Dict]) -> List[Dict]:
        """Extract basic statistical features."""
        features = []
        
        for window_dict in windows:
            sensor_name = window_dict['name']
            window_data = window_dict['data']
            
            # Skip annotation windows
            if sensor_name == 'annotations':
                continue
            
            sensor_features = {'name': sensor_name, 'features': {}}
            
            # Extract statistical features for each window
            means = []
            stds = []
            maxs = []
            mins = []
            ranges = []
            
            for window in window_data:
                # Ensure window is numpy array
                if hasattr(window, 'values'):
                    window = window.values
                
                means.append(np.mean(window))
                stds.append(np.std(window))
                maxs.append(np.max(window))
                mins.append(np.min(window))
                ranges.append(np.max(window) - np.min(window))
            
            # Store features
            sensor_features['features'] = {
                'vgrf_mean': means,
                'vgrf_std': stds,
                'vgrf_max': maxs,
                'vgrf_min': mins,
                'vgrf_range': ranges
            }
            
            features.append(sensor_features)
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features."""
        feature_names = []
        feature_names.extend(self.lbp_extractor.get_feature_names())
        feature_names.extend(self.fourier_extractor.get_feature_names())
        feature_names.extend(['vgrf_mean', 'vgrf_std', 'vgrf_max', 'vgrf_min', 'vgrf_range'])
        return feature_names


# Legacy functions for backward compatibility
def extract_lbp_features(windows: List[Dict], fs: int, **kwargs) -> List[Dict]:
    """
    Legacy function to extract LBP features.
    
    Args:
        windows: List of sliding window dictionaries
        fs: Sampling frequency
        **kwargs: Additional arguments
        
    Returns:
        List of feature dictionaries
    """
    extractor = LBPFeatureExtractor(verbose=kwargs.get('verbose', True))
    return extractor.extract_features(windows, fs, **kwargs)


def extract_fourier_features(windows: List[Dict], fs: int, **kwargs) -> List[Dict]:
    """
    Legacy function to extract Fourier series features.
    
    Args:
        windows: List of sliding window dictionaries
        fs: Sampling frequency
        **kwargs: Additional arguments
        
    Returns:
        List of feature dictionaries
    """
    extractor = FourierSeriesFeatureExtractor(verbose=kwargs.get('verbose', True))
    return extractor.extract_features(windows, fs, **kwargs)


def extract_physionet_features(windows: List[Dict], fs: int, **kwargs) -> List[Dict]:
    """
    Legacy function to extract combined PhysioNet features.
    
    Args:
        windows: List of sliding window dictionaries
        fs: Sampling frequency
        **kwargs: Additional arguments
        
    Returns:
        List of feature dictionaries
    """
    extractor = PhysioNetFeatureExtractor(verbose=kwargs.get('verbose', True))
    return extractor.extract_features(windows, fs, **kwargs) 