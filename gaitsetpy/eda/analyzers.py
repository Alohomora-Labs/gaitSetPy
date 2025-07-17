'''
EDA analyzer classes for gait data analysis.

This module contains individual EDA analyzer classes that inherit from BaseEDAAnalyzer
and provide specific analysis and visualization functionality.

Maintainer: @aharshit123456
'''

from typing import Dict, List, Any, Union, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ..core.base_classes import BaseEDAAnalyzer


class DaphnetVisualizationAnalyzer(BaseEDAAnalyzer):
    """
    EDA analyzer for Daphnet dataset visualization.
    
    This analyzer provides comprehensive visualization capabilities for Daphnet dataset
    including thigh, shank, and trunk sensor data.
    """
    
    def __init__(self):
        super().__init__(
            name="daphnet_visualization",
            description="Comprehensive visualization analyzer for Daphnet dataset sensor data"
        )
        self.config = {
            'figsize': (20, 16),
            'colors': {
                'no_freeze': 'orange',
                'freeze': 'purple'
            },
            'alpha': 0.6
        }
    
    def analyze(self, data: Union[pd.DataFrame, List[pd.DataFrame]], **kwargs) -> Dict[str, Any]:
        """
        Analyze the data and return statistical summaries.
        
        Args:
            data: Input data to analyze
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing analysis results
        """
        if isinstance(data, list):
            # Multiple datasets
            results = {}
            for i, df in enumerate(data):
                results[f'dataset_{i}'] = self._analyze_single_dataset(df)
            return results
        else:
            # Single dataset
            return self._analyze_single_dataset(data)
    
    def _analyze_single_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze a single dataset."""
        # Basic statistics
        stats = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'annotation_distribution': df['annotations'].value_counts().to_dict() if 'annotations' in df.columns else {},
            'missing_values': df.isnull().sum().to_dict(),
            'data_range': {
                'min': df.select_dtypes(include=[np.number]).min().to_dict(),
                'max': df.select_dtypes(include=[np.number]).max().to_dict()
            }
        }
        
        # Sensor-specific statistics
        sensor_stats = {}
        for sensor in ['thigh', 'shank', 'trunk']:
            if sensor in df.columns:
                sensor_stats[sensor] = {
                    'mean': df[sensor].mean(),
                    'std': df[sensor].std(),
                    'min': df[sensor].min(),
                    'max': df[sensor].max()
                }
        
        stats['sensor_statistics'] = sensor_stats
        return stats
    
    def visualize(self, data: Union[pd.DataFrame, List[pd.DataFrame]], **kwargs):
        """
        Create visualizations of the data.
        
        Args:
            data: Input data to visualize
            **kwargs: Additional arguments including sensor_type, dataset_index, names
        """
        sensor_type = kwargs.get('sensor_type', 'all')
        dataset_index = kwargs.get('dataset_index', 0)
        names = kwargs.get('names', [])
        
        if isinstance(data, list):
            if dataset_index < len(data):
                df = data[dataset_index]
                dataset_name = names[dataset_index] if dataset_index < len(names) else f"Dataset {dataset_index}"
            else:
                print(f"Dataset index {dataset_index} out of range")
                return
        else:
            df = data
            dataset_name = names[0] if names else "Dataset"
        
        if sensor_type == 'all':
            self._plot_all_sensors(df, dataset_name)
        elif sensor_type == 'thigh':
            self._plot_thigh_data(df, dataset_name)
        elif sensor_type == 'shank':
            self._plot_shank_data(df, dataset_name)
        elif sensor_type == 'trunk':
            self._plot_trunk_data(df, dataset_name)
        else:
            print(f"Unknown sensor type: {sensor_type}")
    
    def _plot_thigh_data(self, df: pd.DataFrame, dataset_name: str):
        """Plot thigh sensor data."""
        print(f"Plotting thigh data for {dataset_name}")
        
        # Filter data
        df_filtered = df[df.annotations > 0] if 'annotations' in df.columns else df
        
        if df_filtered.empty:
            print("No valid data to plot")
            return
        
        # Create figure
        fig, axes = plt.subplots(4, 1, sharex=True, figsize=self.config['figsize'])
        fig.suptitle(f"Thigh Data from {dataset_name}")
        
        # Separate freeze and no-freeze data
        if 'annotations' in df.columns:
            neg = df_filtered[df_filtered.annotations == 1]  # No freeze
            pos = df_filtered[df_filtered.annotations == 2]  # Freeze
        else:
            neg = df_filtered
            pos = pd.DataFrame()
        
        # Plot each component
        components = ['thigh_h_fd', 'thigh_v', 'thigh_h_l', 'thigh']
        labels = ['Horizontal Forward', 'Vertical', 'Horizontal Lateral', 'Overall']
        
        for i, (component, label) in enumerate(zip(components, labels)):
            if component in df_filtered.columns:
                # Plot main signal
                axes[i].plot(df_filtered.index, df_filtered[component])
                axes[i].set_ylabel(f"{label} Thigh Acceleration")
                
                # Plot annotations if available
                if not neg.empty:
                    axes[i].scatter(neg.index, neg[component], 
                                  c=self.config['colors']['no_freeze'], 
                                  label="no freeze", alpha=self.config['alpha'])
                if not pos.empty:
                    axes[i].scatter(pos.index, pos[component], 
                                  c=self.config['colors']['freeze'], 
                                  label="freeze", alpha=self.config['alpha'])
                
                axes[i].legend()
        
        plt.xlabel("Time")
        plt.tight_layout()
        plt.show()
    
    def _plot_shank_data(self, df: pd.DataFrame, dataset_name: str):
        """Plot shank sensor data."""
        print(f"Plotting shank data for {dataset_name}")
        
        # Filter data
        df_filtered = df[df.annotations > 0] if 'annotations' in df.columns else df
        
        if df_filtered.empty:
            print("No valid data to plot")
            return
        
        # Create figure
        fig, axes = plt.subplots(4, 1, sharex=True, figsize=self.config['figsize'])
        fig.suptitle(f"Shank Data from {dataset_name}")
        
        # Separate freeze and no-freeze data
        if 'annotations' in df.columns:
            neg = df_filtered[df_filtered.annotations == 1]  # No freeze
            pos = df_filtered[df_filtered.annotations == 2]  # Freeze
        else:
            neg = df_filtered
            pos = pd.DataFrame()
        
        # Plot each component
        components = ['shank_h_fd', 'shank_v', 'shank_h_l', 'shank']
        labels = ['Horizontal Forward', 'Vertical', 'Horizontal Lateral', 'Overall']
        
        for i, (component, label) in enumerate(zip(components, labels)):
            if component in df_filtered.columns:
                # Plot main signal
                axes[i].plot(df_filtered.index, df_filtered[component])
                axes[i].set_ylabel(f"{label} Shank Acceleration")
                
                # Plot annotations if available
                if not neg.empty:
                    axes[i].scatter(neg.index, neg[component], 
                                  c=self.config['colors']['no_freeze'], 
                                  label="no freeze", alpha=self.config['alpha'])
                if not pos.empty:
                    axes[i].scatter(pos.index, pos[component], 
                                  c=self.config['colors']['freeze'], 
                                  label="freeze", alpha=self.config['alpha'])
                
                axes[i].legend()
        
        plt.xlabel("Time")
        plt.tight_layout()
        plt.show()
    
    def _plot_trunk_data(self, df: pd.DataFrame, dataset_name: str):
        """Plot trunk sensor data."""
        print(f"Plotting trunk data for {dataset_name}")
        
        # Filter data
        df_filtered = df[df.annotations > 0] if 'annotations' in df.columns else df
        
        if df_filtered.empty:
            print("No valid data to plot")
            return
        
        # Create figure
        fig, axes = plt.subplots(4, 1, sharex=True, figsize=self.config['figsize'])
        fig.suptitle(f"Trunk Data from {dataset_name}")
        
        # Separate freeze and no-freeze data
        if 'annotations' in df.columns:
            neg = df_filtered[df_filtered.annotations == 1]  # No freeze
            pos = df_filtered[df_filtered.annotations == 2]  # Freeze
        else:
            neg = df_filtered
            pos = pd.DataFrame()
        
        # Plot each component
        components = ['trunk_h_fd', 'trunk_v', 'trunk_h_l', 'trunk']
        labels = ['Horizontal Forward', 'Vertical', 'Horizontal Lateral', 'Overall']
        
        for i, (component, label) in enumerate(zip(components, labels)):
            if component in df_filtered.columns:
                # Plot main signal
                axes[i].plot(df_filtered.index, df_filtered[component])
                axes[i].set_ylabel(f"{label} Trunk Acceleration")
                
                # Plot annotations if available
                if not neg.empty:
                    axes[i].scatter(neg.index, neg[component], 
                                  c=self.config['colors']['no_freeze'], 
                                  label="no freeze", alpha=self.config['alpha'])
                if not pos.empty:
                    axes[i].scatter(pos.index, pos[component], 
                                  c=self.config['colors']['freeze'], 
                                  label="freeze", alpha=self.config['alpha'])
                
                axes[i].legend()
        
        plt.xlabel("Time")
        plt.tight_layout()
        plt.show()
    
    def _plot_all_sensors(self, df: pd.DataFrame, dataset_name: str):
        """Plot all sensor data in a combined view."""
        print(f"Plotting all sensor data for {dataset_name}")
        
        # Create figure with subplots for each sensor
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=self.config['figsize'])
        fig.suptitle(f"All Sensor Data from {dataset_name}")
        
        # Filter data
        df_filtered = df[df.annotations > 0] if 'annotations' in df.columns else df
        
        if df_filtered.empty:
            print("No valid data to plot")
            return
        
        sensors = ['thigh', 'shank', 'trunk']
        for i, sensor in enumerate(sensors):
            if sensor in df_filtered.columns:
                axes[i].plot(df_filtered.index, df_filtered[sensor])
                axes[i].set_ylabel(f"{sensor.capitalize()} Acceleration")
                
                # Add annotations if available
                if 'annotations' in df_filtered.columns:
                    neg = df_filtered[df_filtered.annotations == 1]
                    pos = df_filtered[df_filtered.annotations == 2]
                    
                    if not neg.empty:
                        axes[i].scatter(neg.index, neg[sensor], 
                                      c=self.config['colors']['no_freeze'], 
                                      label="no freeze", alpha=self.config['alpha'])
                    if not pos.empty:
                        axes[i].scatter(pos.index, pos[sensor], 
                                      c=self.config['colors']['freeze'], 
                                      label="freeze", alpha=self.config['alpha'])
                    
                    axes[i].legend()
        
        plt.xlabel("Time")
        plt.tight_layout()
        plt.show()


class SensorStatisticsAnalyzer(BaseEDAAnalyzer):
    """
    EDA analyzer for sensor data statistics and feature visualization.
    
    This analyzer provides statistical analysis and feature visualization capabilities
    for sensor data including sliding windows and extracted features.
    """
    
    def __init__(self):
        super().__init__(
            name="sensor_statistics",
            description="Statistical analysis and feature visualization for sensor data"
        )
        self.config = {
            'figsize': (20, 10),
            'feature_markers': {
                'mean': 'x',
                'rms': 'o',
                'peak_height': 'v',
                'mode': '<',
                'median': '^'
            }
        }
    
    def analyze(self, data: Union[pd.DataFrame, List[pd.DataFrame]], **kwargs) -> Dict[str, Any]:
        """
        Analyze sensor data and return statistical summaries.
        
        Args:
            data: Input data to analyze
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing analysis results
        """
        if isinstance(data, list):
            # Multiple datasets
            results = {}
            for i, df in enumerate(data):
                results[f'dataset_{i}'] = self._compute_statistics(df)
            return results
        else:
            # Single dataset
            return self._compute_statistics(data)
    
    def _compute_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute comprehensive statistics for a dataset."""
        stats = {
            'basic_stats': df.describe().to_dict(),
            'correlation_matrix': df.corr().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 1 else {},
            'skewness': df.skew().to_dict(),
            'kurtosis': df.kurtosis().to_dict()
        }
        
        # Add sensor-specific statistics
        sensor_stats = {}
        for sensor in ['thigh', 'shank', 'trunk']:
            if sensor in df.columns:
                sensor_data = df[sensor].dropna()
                sensor_stats[sensor] = {
                    'mean': sensor_data.mean(),
                    'std': sensor_data.std(),
                    'variance': sensor_data.var(),
                    'min': sensor_data.min(),
                    'max': sensor_data.max(),
                    'range': sensor_data.max() - sensor_data.min(),
                    'median': sensor_data.median(),
                    'q25': sensor_data.quantile(0.25),
                    'q75': sensor_data.quantile(0.75),
                    'iqr': sensor_data.quantile(0.75) - sensor_data.quantile(0.25)
                }
        
        stats['sensor_statistics'] = sensor_stats
        return stats
    
    def visualize(self, sliding_windows: List[Dict], features: List[Dict], **kwargs):
        """
        Create visualizations of sensor data with overlaid features.
        
        Args:
            sliding_windows: List of sliding window dictionaries
            features: List of feature dictionaries
            **kwargs: Additional arguments including sensor_name, start_idx, end_idx, num_windows
        """
        sensor_name = kwargs.get('sensor_name', 'shank')
        start_idx = kwargs.get('start_idx', 0)
        end_idx = kwargs.get('end_idx', 1000)
        num_windows = kwargs.get('num_windows', 10)
        save = kwargs.get('save', False)
        
        self._plot_sensor_with_features(sliding_windows, features, start_idx, end_idx, 
                                      sensor_name, num_windows, save)
    
    def _plot_sensor_with_features(self, sliding_windows: List[Dict], features: List[Dict], 
                                 start_idx: int, end_idx: int, sensor_name: str = "shank", 
                                 num_windows: int = 10, save: bool = False):
        """
        Plot sliding windows of sensor data with overlaid statistical features.
        
        Args:
            sliding_windows: List of sliding window dictionaries
            features: List of feature dictionaries
            start_idx: Start index of the time window
            end_idx: End index of the time window
            sensor_name: Name of the sensor to plot
            num_windows: Number of sliding windows to plot
            save: Whether to save the plot
        """
        fig, axes = plt.subplots(2, 1, figsize=self.config['figsize'], 
                                gridspec_kw={'height_ratios': [3, 1]})
        
        # Extract sensor windows
        sensor_windows = next((sw['data'] for sw in sliding_windows if sw['name'] == sensor_name), None)
        if sensor_windows is None:
            print(f"Sensor '{sensor_name}' not found in sliding_windows.")
            return
        
        # Extract corresponding features
        sensor_features = next((feat['features'] for feat in features if feat['name'] == sensor_name), None)
        if sensor_features is None:
            print(f"Sensor '{sensor_name}' not found in features.")
            return
        
        # Filter windows based on start_idx and end_idx
        filtered_windows = [series for series in sensor_windows 
                           if start_idx <= series.index[0] and series.index[-1] <= end_idx]
        
        if not filtered_windows:
            print(f"No windows found in the specified index range ({start_idx} - {end_idx}).")
            return
        
        # Store entropy & frequency features for separate plotting
        entropy_values = []
        dominant_frequencies = []
        
        # Plot first num_windows windows
        for i in range(min(num_windows, len(filtered_windows))):
            series = filtered_windows[i]
            
            # Extract time and signal values
            time_values = series.index.to_numpy()
            signal_values = series.values
            
            # Determine actual start and end indices for this window
            window_start, window_end = time_values[0], time_values[-1]
            
            # Plot time series data
            axes[0].plot(time_values, signal_values, alpha=0.6)
            
            # Mark start and end of each window with vertical dotted lines
            axes[0].axvline(x=window_start, color='black', linestyle='dotted', alpha=0.7)
            axes[0].axvline(x=window_end, color='black', linestyle='dotted', alpha=0.7)
            
            # Overlay statistical features
            for feature_name, marker in self.config['feature_markers'].items():
                if feature_name in sensor_features and len(sensor_features[feature_name]) > i:
                    feature_value = sensor_features[feature_name][i]
                    if feature_value != 0:  # Skip zero values
                        closest_index = np.argmin(np.abs(signal_values - feature_value))
                        closest_time = time_values[closest_index]
                        axes[0].scatter(closest_time, feature_value, color='red', 
                                      marker=marker, s=100, label=feature_name if i == 0 else "")
            
            # Store entropy & frequency features for separate plotting
            if 'entropy' in sensor_features and len(sensor_features['entropy']) > i:
                entropy_values.append(sensor_features['entropy'][i])
            if 'dominant_frequency' in sensor_features and len(sensor_features['dominant_frequency']) > i:
                dominant_frequencies.append(sensor_features['dominant_frequency'][i])
        
        # Labels and title for time-series plot
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel(f'{sensor_name} Signal')
        axes[0].set_title(f'First {num_windows} windows of {sensor_name} in range {start_idx}-{end_idx} with Features')
        axes[0].legend()
        
        # Frequency-domain & entropy plot
        if dominant_frequencies:
            window_indices = list(range(len(dominant_frequencies)))
            axes[1].plot(window_indices, dominant_frequencies, 
                        label="Dominant Frequency", marker="o", linestyle="dashed", color="blue")
        
        if entropy_values:
            axes[1].bar(window_indices, entropy_values, alpha=0.6, label="Entropy", color="green")
        
        axes[1].set_xlabel("Window Index")
        axes[1].set_ylabel("Feature Value")
        axes[1].set_title("Frequency & Entropy Features")
        axes[1].legend()
        
        plt.tight_layout()
        
        # Save or show plot
        if save:
            file_path = input("Enter the file path to save the plot (e.g., 'plot.png'): ")
            plt.savefig(file_path, dpi=300)
            print(f"Plot saved at {file_path}")
        else:
            plt.show() 


def harup_basic_stats(harup_df):
    """
    Print and return basic statistics for each sensor column in a HAR-UP DataFrame.
    Args:
        harup_df (pd.DataFrame): DataFrame containing HAR-UP data.
    Returns:
        pd.DataFrame: DataFrame of statistics.
    """
    import pandas as pd
    stats = harup_df.describe().T
    print(stats)
    return stats

def harup_missing_data_report(harup_df):
    """
    Print and return missing value counts for each column in a HAR-UP DataFrame.
    Args:
        harup_df (pd.DataFrame): DataFrame containing HAR-UP data.
    Returns:
        pd.Series: Series of missing value counts.
    """
    missing = harup_df.isnull().sum()
    print(missing)
    return missing

def harup_activity_stats(harup_df):
    """
    Print and return counts for each activity label in a HAR-UP DataFrame.
    Args:
        harup_df (pd.DataFrame): DataFrame containing HAR-UP data.
    Returns:
        pd.Series: Series of activity label counts.
    """
    if 'activity_label' not in harup_df.columns:
        print("No 'activity_label' column found.")
        return None
    counts = harup_df['activity_label'].value_counts().sort_index()
    print(counts)
    return counts 