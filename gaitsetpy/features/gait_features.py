'''
Gait Feature Extractor Class
Maintainer: @aharshit123456

This module contains the GaitFeatureExtractor class that inherits from BaseFeatureExtractor
and provides comprehensive gait feature extraction functionality.
'''

from typing import List, Dict, Any
import numpy as np
import logging
from tqdm import tqdm
from ..core.base_classes import BaseFeatureExtractor
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
    calculate_energy,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GaitFeatureExtractor(BaseFeatureExtractor):
    """
    Comprehensive gait feature extractor class.
    
    This class extracts various time-domain, frequency-domain, and statistical features
    from gait data sliding windows.
    """
    
    def __init__(self, verbose: bool = True):
        super().__init__(
            name="gait_features",
            description="Comprehensive gait feature extractor for time-domain, frequency-domain, and statistical features"
        )
        self.verbose = verbose
        self.config = {
            'time_domain': True,
            'frequency_domain': True,
            'statistical': True,
            'ar_order': 3  # Order for auto-regression coefficients
        }
        
        if self.verbose:
            print("🚀 GaitFeatureExtractor initialized successfully!")
            print(f"📊 Default configuration: {self.config}")
    
    def extract_features(self, windows: List[Dict], fs: int, **kwargs) -> List[Dict]:
        """
        Extract gait features from sliding windows.
        
        Args:
            windows: List of sliding window dictionaries
            fs: Sampling frequency
            **kwargs: Additional arguments including time_domain, frequency_domain, statistical flags
            
        Returns:
            List of feature dictionaries for each sensor
        """
        # Update config with any passed arguments
        time_domain = kwargs.get('time_domain', self.config['time_domain'])
        frequency_domain = kwargs.get('frequency_domain', self.config['frequency_domain'])
        statistical = kwargs.get('statistical', self.config['statistical'])
        ar_order = kwargs.get('ar_order', self.config['ar_order'])
        
        if self.verbose:
            print("\n" + "="*60)
            print("🔍 STARTING GAIT FEATURE EXTRACTION")
            print("="*60)
            print(f"📈 Total sensors/windows to process: {len(windows)}")
            print(f"🔊 Sampling frequency: {fs} Hz")
            print(f"⏱️  Time domain features: {'✅' if time_domain else '❌'}")
            print(f"🌊 Frequency domain features: {'✅' if frequency_domain else '❌'}")
            print(f"📊 Statistical features: {'✅' if statistical else '❌'}")
            print(f"🔄 Auto-regression order: {ar_order}")
            print("-"*60)
        
        features = []
        
        # Main progress bar for processing all windows
        main_pbar = tqdm(
            windows, 
            desc="🔍 Processing Sensors", 
            unit="sensor",
            disable=not self.verbose
        )
        
        for i, window_dict in enumerate(main_pbar):
            sensor_name = window_dict['name']
            window_data = window_dict['data']
            
            if self.verbose:
                main_pbar.set_postfix({
                    'Current': sensor_name,
                    'Windows': len(window_data) if isinstance(window_data, list) else 1
                })
            
            # Skip annotation windows
            if sensor_name == 'annotations':
                if self.verbose:
                    logger.info(f"📝 Processing annotation data for {sensor_name}")
                
                features.append({
                    'name': sensor_name,
                    'features': {},
                    'annotations': [self._extract_annotation_labels(window) for window in window_data]
                })
                continue
            
            if self.verbose:
                logger.info(f"🎯 Processing sensor: {sensor_name}")
                logger.info(f"📦 Number of windows: {len(window_data)}")
            
            sensor_features = {'name': sensor_name, 'features': {}}
            
            # Time domain features
            if time_domain:
                if self.verbose:
                    print(f"  ⏱️  Extracting time domain features for {sensor_name}...")
                
                time_features = self._extract_time_domain_features(window_data)
                sensor_features['features'].update(time_features)
                
                if self.verbose:
                    feature_count = sum(len(v) if isinstance(v, list) else 1 for v in time_features.values())
                    print(f"  ✅ Time domain: {len(time_features)} feature types, {feature_count} total features")
            
            # Frequency domain features
            if frequency_domain:
                if self.verbose:
                    print(f"  🌊 Extracting frequency domain features for {sensor_name}...")
                
                freq_features = self._extract_frequency_domain_features(window_data, fs)
                sensor_features['features'].update(freq_features)
                
                if self.verbose:
                    feature_count = sum(len(v) if isinstance(v, list) else 1 for v in freq_features.values())
                    print(f"  ✅ Frequency domain: {len(freq_features)} feature types, {feature_count} total features")
            
            # Statistical features
            if statistical:
                if self.verbose:
                    print(f"  📊 Extracting statistical features for {sensor_name}...")
                
                stat_features = self._extract_statistical_features(window_data)
                sensor_features['features'].update(stat_features)
                
                if self.verbose:
                    feature_count = sum(len(v) if isinstance(v, list) else 1 for v in stat_features.values())
                    print(f"  ✅ Statistical: {len(stat_features)} feature types, {feature_count} total features")
            
            # Auto-regression coefficients
            if self.verbose:
                print(f"  🔄 Extracting auto-regression coefficients for {sensor_name}...")
            
            ar_features = self._extract_ar_coefficients(window_data, ar_order)
            sensor_features['features'].update(ar_features)
            
            if self.verbose:
                feature_count = sum(len(v) if isinstance(v, list) else 1 for v in ar_features.values())
                print(f"  ✅ Auto-regression: {len(ar_features)} feature types, {feature_count} total features")
            
            # Calculate total features for this sensor
            total_features = sum(
                len(v) if isinstance(v, list) else 1 
                for v in sensor_features['features'].values()
            )
            
            if self.verbose:
                print(f"  🎯 Total features extracted for {sensor_name}: {total_features}")
                print(f"  📋 Feature types: {list(sensor_features['features'].keys())}")
                print("-"*40)
            
            features.append(sensor_features)
        
        if self.verbose:
            print("\n" + "="*60)
            print("🎉 FEATURE EXTRACTION COMPLETED!")
            print("="*60)
            print(f"📊 Total sensors processed: {len(features)}")
            
            # Calculate overall statistics
            total_feature_count = 0
            for feature_dict in features:
                if 'features' in feature_dict:
                    total_feature_count += sum(
                        len(v) if isinstance(v, list) else 1 
                        for v in feature_dict['features'].values()
                    )
            
            print(f"🔢 Total features extracted: {total_feature_count}")
            print(f"📈 Average features per sensor: {total_feature_count / len(features):.1f}")
            print("="*60)
        
        return features
    
    def _extract_time_domain_features(self, windows: List) -> Dict[str, List]:
        """Extract time domain features from windows."""
        if self.verbose:
            print("    🔍 Computing time domain features...")
        
        time_features = {}
        
        # Define time domain feature functions
        time_domain_funcs = {
            'mean': calculate_mean,
            'std': calculate_standard_deviation,
            'variance': calculate_variance,
            'rms': calculate_root_mean_square,
            'range': calculate_range,
            'median': calculate_median,
            'mode': calculate_mode,
            'mean_absolute_value': calculate_mean_absolute_value,
            'median_absolute_deviation': calculate_median_absolute_deviation,
            'peak_height': calculate_peak_height,
            'zero_crossing_rate': calculate_zero_crossing_rate,
            'energy': calculate_energy,
        }
        
        # Progress bar for time domain features
        feature_pbar = tqdm(
            time_domain_funcs.items(), 
            desc="    ⏱️  Time features", 
            unit="feature",
            leave=False,
            disable=not self.verbose
        )
        
        for feature_name, func in feature_pbar:
            if self.verbose:
                feature_pbar.set_postfix({'Computing': feature_name})
            
            time_features[feature_name] = [
                func(self._ensure_numpy_array(window)) for window in windows
            ]
        
        return time_features
    
    def _ensure_numpy_array(self, signal):
        """Convert pandas Series to numpy array if needed."""
        if hasattr(signal, 'values'):
            return signal.values
        return signal
    
    def _extract_frequency_domain_features(self, windows: List, fs: int) -> Dict[str, List]:
        """Extract frequency domain features from windows."""
        if self.verbose:
            print("    🔍 Computing frequency domain features...")
        
        freq_features = {}
        
        # Define frequency domain feature functions
        freq_domain_funcs = {
            'dominant_frequency': lambda w: calculate_dominant_frequency(w, fs),
            'peak_frequency': lambda w: calculate_peak_frequency(w, fs),
            'power_spectral_entropy': lambda w: calculate_power_spectral_entropy(w, fs),
            'principal_harmonic_frequency': lambda w: calculate_principal_harmonic_frequency(w, fs),
            'stride_times': lambda w: calculate_stride_times(w, fs),
            'step_time': lambda w: calculate_step_time(w, fs),
            'cadence': lambda w: calculate_cadence(w, fs),
            'freezing_index': lambda w: calculate_freezing_index(w, fs),
        }
        
        # Progress bar for frequency domain features
        feature_pbar = tqdm(
            freq_domain_funcs.items(), 
            desc="    🌊 Freq features", 
            unit="feature",
            leave=False,
            disable=not self.verbose
        )
        
        for feature_name, func in feature_pbar:
            if self.verbose:
                feature_pbar.set_postfix({'Computing': feature_name})
            
            freq_features[feature_name] = [
                func(self._ensure_numpy_array(window)) for window in windows
            ]
        
        return freq_features
    
    def _extract_statistical_features(self, windows: List) -> Dict[str, List]:
        """Extract statistical features from windows."""
        if self.verbose:
            print("    🔍 Computing statistical features...")
        
        stat_features = {}
        
        # Define statistical feature functions
        stat_funcs = {
            'skewness': calculate_skewness,
            'kurtosis': calculate_kurtosis,
            'entropy': calculate_entropy,
            'interquartile_range': calculate_interquartile_range,
        }
        
        # Progress bar for statistical features
        feature_pbar = tqdm(
            stat_funcs.items(), 
            desc="    📊 Stat features", 
            unit="feature",
            leave=False,
            disable=not self.verbose
        )
        
        for feature_name, func in feature_pbar:
            if self.verbose:
                feature_pbar.set_postfix({'Computing': feature_name})
            
            stat_features[feature_name] = [
                func(self._ensure_numpy_array(window)) for window in windows
            ]
        
        # Handle correlation separately (needs two signals)
        if self.verbose:
            print("      🔗 Computing correlation features...")
        
        stat_features['correlation'] = [
            calculate_correlation(
                self._ensure_numpy_array(window)[:-1], 
                self._ensure_numpy_array(window)[1:]
            ) if len(window) > 1 else 0 
            for window in windows
        ]
        
        return stat_features
    
    def _extract_ar_coefficients(self, windows: List, order: int) -> Dict[str, List]:
        """Extract auto-regression coefficients from windows."""
        if self.verbose:
            print(f"    🔍 Computing auto-regression coefficients (order={order})...")
        
        # Progress bar for AR coefficients
        ar_pbar = tqdm(
            windows, 
            desc="    🔄 AR coeffs", 
            unit="window",
            leave=False,
            disable=not self.verbose
        )
        
        ar_coeffs = []
        for window in ar_pbar:
            coeffs = calculate_auto_regression_coefficients(
                self._ensure_numpy_array(window), order
            )
            ar_coeffs.append(coeffs)
        
        return {'ar_coefficients': ar_coeffs}
    
    def _extract_annotation_labels(self, window) -> int:
        """Extract the most common annotation label from a window."""
        if hasattr(window, 'mode'):
            return window.mode().iloc[0] if len(window.mode()) > 0 else 0
        else:
            # For numpy arrays or other types
            unique, counts = np.unique(window, return_counts=True)
            return unique[np.argmax(counts)]
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of all features that can be extracted.
        
        Returns:
            List of feature names
        """
        time_domain_features = [
            'mean', 'std', 'variance', 'rms', 'range', 'median', 'mode',
            'mean_absolute_value', 'median_absolute_deviation', 'peak_height',
            'zero_crossing_rate', 'energy'
        ]
        
        frequency_domain_features = [
            'dominant_frequency', 'peak_frequency', 'power_spectral_entropy',
            'principal_harmonic_frequency', 'stride_times', 'step_time',
            'cadence', 'freezing_index'
        ]
        
        statistical_features = [
            'skewness', 'kurtosis', 'entropy', 'interquartile_range', 'correlation'
        ]
        
        other_features = ['ar_coefficients']
        
        return time_domain_features + frequency_domain_features + statistical_features + other_features

    def print_extraction_summary(self, features: List[Dict]) -> None:
        """
        Print a detailed summary of extracted features.
        
        Args:
            features: List of feature dictionaries returned by extract_features
        """
        print("\n" + "="*80)
        print("📊 FEATURE EXTRACTION SUMMARY")
        print("="*80)
        
        for i, feature_dict in enumerate(features):
            sensor_name = feature_dict['name']
            print(f"\n🎯 Sensor {i+1}: {sensor_name}")
            print("-" * 40)
            
            if 'features' in feature_dict and feature_dict['features']:
                for feature_type, feature_values in feature_dict['features'].items():
                    if isinstance(feature_values, list):
                        print(f"  📈 {feature_type}: {len(feature_values)} values")
                        if feature_values:
                            sample_value = feature_values[0]
                            if isinstance(sample_value, (list, np.ndarray)):
                                print(f"    └── Shape per window: {np.array(sample_value).shape}")
                            else:
                                print(f"    └── Sample value: {sample_value:.4f}")
                    else:
                        print(f"  📈 {feature_type}: {feature_values}")
            
            if 'annotations' in feature_dict:
                print(f"  📝 Annotations: {len(feature_dict['annotations'])} windows")
        
        print("\n" + "="*80)


# Legacy function wrappers for backward compatibility
def get_stride_times_for_windows(windows, fs):
    """Calculate stride times for all windows in the input."""
    return [calculate_stride_times(window, fs) for window in windows]

def get_zero_crossing_rates_for_windows(windows):
    """Calculate zero-crossing rates for all windows in the input."""
    return [calculate_zero_crossing_rate(window) for window in windows]

def get_freezing_indices_for_windows(windows, fs):
    """Calculate freezing indices for all windows in the input."""
    return [calculate_freezing_index(window, fs) for window in windows]

def get_standard_deviations_for_windows(windows):
    """Calculate standard deviations for all windows in the input."""
    return [calculate_standard_deviation(window) for window in windows]

def get_entropies_for_windows(windows):
    """Calculate entropies for all windows in the input."""
    return [calculate_entropy(window) for window in windows]

def get_energies_for_windows(windows):
    """Calculate energies for all windows in the input."""
    return [calculate_energy(window) for window in windows]

def get_variances_for_windows(windows):
    """Calculate variances for all windows in the input."""
    return [calculate_variance(window) for window in windows]

def get_kurtosis_for_windows(windows):
    """Calculate kurtosis values for all windows in the input."""
    return [calculate_kurtosis(window) for window in windows]

def get_step_times_for_windows(windows, fs):
    """Calculate step times for all windows in the input."""
    return [calculate_step_time(window, fs) for window in windows]

def get_mean_for_windows(windows):
    return [calculate_mean(window) for window in windows]

def get_standard_deviation_for_windows(windows):
    return [calculate_standard_deviation(window) for window in windows]

def get_variance_for_windows(windows):
    return [calculate_variance(window) for window in windows]

def get_skewness_for_windows(windows):
    return [calculate_skewness(window) for window in windows]

def get_root_mean_square_for_windows(windows):
    return [calculate_root_mean_square(window) for window in windows]

def get_range_for_windows(windows):
    return [calculate_range(window) for window in windows]

def get_median_for_windows(windows):
    return [calculate_median(window) for window in windows]

def get_mode_for_windows(windows):
    return [calculate_mode(window) for window in windows]

def get_mean_absolute_value_for_windows(windows):
    return [calculate_mean_absolute_value(window) for window in windows]

def get_median_absolute_deviation_for_windows(windows):
    return [calculate_median_absolute_deviation(window) for window in windows]

def get_peak_height_for_windows(windows):
    return [calculate_peak_height(window) for window in windows]

def get_cadence_for_windows(windows, fs):
    return [calculate_cadence(window, fs) for window in windows]

def get_freezing_index_for_windows(windows, fs):
    return [calculate_freezing_index(window, fs) for window in windows]

def get_dominant_frequency_for_windows(windows, fs):
    return [calculate_dominant_frequency(window, fs) for window in windows]

def get_peak_frequency_for_windows(windows, fs):
    return [calculate_peak_frequency(window, fs) for window in windows]

def get_power_spectral_entropy_for_windows(windows, fs):
    return [calculate_power_spectral_entropy(window, fs) for window in windows]

def get_principal_harmonic_frequency_for_windows(windows, fs):
    return [calculate_principal_harmonic_frequency(window, fs) for window in windows]

def get_entropy_for_windows(windows):
    return [calculate_entropy(window) for window in windows]

def get_interquartile_range_for_windows(windows):
    return [calculate_interquartile_range(window) for window in windows]

def get_correlation_for_windows(windows):
    # For correlation, we need to handle it differently since it needs two signals
    # We'll calculate autocorrelation for each window
    return [calculate_correlation(window[:-1], window[1:]) if len(window) > 1 else 0 for window in windows]

def get_auto_regression_coefficients_for_windows(windows, order=3):
    return [calculate_auto_regression_coefficients(window, order) for window in windows]

def extract_gait_features(daphnet_windows, fs, time_domain=True, frequency_domain=True, statistical=True, verbose=True):
    """
    Legacy function for extracting gait features.
    
    Args:
        daphnet_windows: List of sliding window dictionaries
        fs: Sampling frequency
        time_domain: Whether to extract time domain features
        frequency_domain: Whether to extract frequency domain features
        statistical: Whether to extract statistical features
        verbose: Whether to show verbose output and progress bars
        
    Returns:
        List of feature dictionaries
    """
    extractor = GaitFeatureExtractor(verbose=verbose)
    return extractor.extract_features(
        daphnet_windows, fs, 
        time_domain=time_domain, 
        frequency_domain=frequency_domain, 
        statistical=statistical
    )
