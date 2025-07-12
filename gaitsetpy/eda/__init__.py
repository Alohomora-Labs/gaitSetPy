"""
eda: Provides exploratory data analysis tools for gait datasets.

This module provides both the new class-based EDA analyzers and legacy function-based API.
All EDA analyzers inherit from BaseEDAAnalyzer and are registered with the EDAManager.

Features:
- Comprehensive visualization of sensor data (thigh, shank, trunk)
- Statistical analysis and summaries
- Feature visualization and overlay
- Dataset comparison and analysis

Maintainer: @aharshit123456
"""

# Import the new class-based EDA analyzers
from .analyzers import (
    DaphnetVisualizationAnalyzer,
    SensorStatisticsAnalyzer
)

# Import legacy functions for backward compatibility
from .visualization import (
    plot_thigh_data,
    plot_shank_data,
    plot_trunk_data,
    plot_all_data,
    plot_all_thigh_data,
    plot_all_shank_data,
    plot_all_trunk_data,
    plot_all_datasets
)

from .statistics import plot_sensor_with_features

# Import managers
from ..core.managers import EDAManager

# Register all EDA analyzers with the manager
def _register_analyzers():
    """Register all available EDA analyzers with the EDAManager."""
    manager = EDAManager()
    manager.register_analyzer("daphnet_visualization", DaphnetVisualizationAnalyzer)
    manager.register_analyzer("sensor_statistics", SensorStatisticsAnalyzer)

# Auto-register analyzers when module is imported
_register_analyzers()

# Convenient access to the EDA manager
def get_eda_manager():
    """Get the singleton EDAManager instance."""
    return EDAManager()

# Helper function to get available analyzers
def get_available_analyzers():
    """Get list of available EDA analyzer names."""
    return EDAManager().get_available_components()

# Helper function to analyze data using manager
def analyze_data(analyzer_name: str, data, **kwargs):
    """
    Analyze data using the EDAManager.
    
    Args:
        analyzer_name: Name of the EDA analyzer
        data: Input data to analyze
        **kwargs: Additional arguments for analysis
        
    Returns:
        Analysis results dictionary
    """
    return EDAManager().analyze_data(analyzer_name, data, **kwargs)

# Helper function to visualize data using manager
def visualize_data(analyzer_name: str, data, **kwargs):
    """
    Create visualizations using the EDAManager.
    
    Args:
        analyzer_name: Name of the EDA analyzer
        data: Input data to visualize
        **kwargs: Additional arguments for visualization
    """
    return EDAManager().visualize_data(analyzer_name, data, **kwargs)

# Convenient wrapper functions for common operations
def plot_daphnet_data(data, names=None, sensor_type='all', dataset_index=0):
    """
    Plot Daphnet dataset using the DaphnetVisualizationAnalyzer.
    
    Args:
        data: Input data (DataFrame or list of DataFrames)
        names: List of dataset names
        sensor_type: Type of sensor to plot ('all', 'thigh', 'shank', 'trunk')
        dataset_index: Index of dataset to plot (if data is a list)
    """
    analyzer = DaphnetVisualizationAnalyzer()
    analyzer.visualize(data, sensor_type=sensor_type, dataset_index=dataset_index, names=names or [])

def analyze_sensor_statistics(data):
    """
    Analyze sensor statistics using the SensorStatisticsAnalyzer.
    
    Args:
        data: Input data (DataFrame or list of DataFrames)
        
    Returns:
        Dictionary containing statistical analysis results
    """
    analyzer = SensorStatisticsAnalyzer()
    return analyzer.analyze(data)

def plot_sensor_features(sliding_windows, features, sensor_name='shank', start_idx=0, end_idx=1000, num_windows=10):
    """
    Plot sensor data with overlaid features using the SensorStatisticsAnalyzer.
    
    Args:
        sliding_windows: List of sliding window dictionaries
        features: List of feature dictionaries
        sensor_name: Name of the sensor to plot
        start_idx: Start index of the time window
        end_idx: End index of the time window
        num_windows: Number of sliding windows to plot
    """
    analyzer = SensorStatisticsAnalyzer()
    analyzer.visualize(sliding_windows, features, sensor_name=sensor_name, 
                      start_idx=start_idx, end_idx=end_idx, num_windows=num_windows)

__all__ = [
    # New class-based analyzers
    'DaphnetVisualizationAnalyzer',
    'SensorStatisticsAnalyzer',
    # Legacy functions for backward compatibility
    'plot_thigh_data',
    'plot_shank_data',
    'plot_trunk_data',
    'plot_all_data',
    'plot_all_thigh_data',
    'plot_all_shank_data',
    'plot_all_trunk_data',
    'plot_all_datasets',
    'plot_sensor_with_features',
    # Manager functions
    'get_eda_manager',
    'get_available_analyzers',
    'analyze_data',
    'visualize_data',
    # Convenient wrapper functions
    'plot_daphnet_data',
    'analyze_sensor_statistics',
    'plot_sensor_features'
]