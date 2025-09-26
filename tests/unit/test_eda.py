"""
Unit tests for EDA (Exploratory Data Analysis) functionality in GaitSetPy.

This module tests the EDA analyzers and visualization utilities
for gait data analysis.

Maintainer: @aharshit123456
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, Mock, MagicMock

from gaitsetpy.eda.analyzers import (
    DaphnetVisualizationAnalyzer,
    SensorStatisticsAnalyzer
)


class TestDaphnetVisualizationAnalyzer:
    """Test cases for DaphnetVisualizationAnalyzer."""
    
    def test_instantiation(self):
        """Test DaphnetVisualizationAnalyzer instantiation."""
        analyzer = DaphnetVisualizationAnalyzer()
        
        assert analyzer.name == "daphnet_visualization"
        assert "daphnet" in analyzer.description.lower()
        assert analyzer.config['figsize'] == (20, 16)
        assert 'colors' in analyzer.config
        assert 'alpha' in analyzer.config
    
    def test_analyze_single_dataset(self, sample_daphnet_data):
        """Test analyzing a single dataset."""
        analyzer = DaphnetVisualizationAnalyzer()
        data, names = sample_daphnet_data
        
        result = analyzer.analyze(data[0])
        
        assert isinstance(result, dict)
        assert 'shape' in result
        assert 'columns' in result
        assert 'annotation_distribution' in result
        assert 'missing_values' in result
        assert 'data_range' in result
        assert 'sensor_statistics' in result
    
    def test_analyze_multiple_datasets(self, sample_daphnet_data):
        """Test analyzing multiple datasets."""
        analyzer = DaphnetVisualizationAnalyzer()
        data, names = sample_daphnet_data
        
        result = analyzer.analyze(data)
        
        assert isinstance(result, dict)
        assert 'dataset_0' in result
        assert 'dataset_1' in result
        assert 'dataset_2' in result
    
    def test_analyze_empty_data(self):
        """Test analyzing empty data."""
        analyzer = DaphnetVisualizationAnalyzer()
        
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        result = analyzer.analyze(empty_df)
        
        assert isinstance(result, dict)
        assert result['shape'] == (0, 0)
    
    def test_analyze_data_without_annotations(self):
        """Test analyzing data without annotations column."""
        analyzer = DaphnetVisualizationAnalyzer()
        
        # Create DataFrame without annotations
        df = pd.DataFrame({
            'shank': [1, 2, 3, 4, 5],
            'thigh': [2, 3, 4, 5, 6],
            'trunk': [3, 4, 5, 6, 7]
        })
        
        result = analyzer.analyze(df)
        
        assert isinstance(result, dict)
        assert result['annotation_distribution'] == {}
    
    def test_visualize_all_sensors(self, sample_daphnet_data, mock_matplotlib):
        """Test visualizing all sensors."""
        analyzer = DaphnetVisualizationAnalyzer()
        data, names = sample_daphnet_data
        
        # Should not raise an error
        analyzer.visualize(data, names=names, sensor_type='all')
    
    def test_visualize_thigh_data(self, sample_daphnet_data, mock_matplotlib):
        """Test visualizing thigh data."""
        analyzer = DaphnetVisualizationAnalyzer()
        data, names = sample_daphnet_data
        
        # Should not raise an error
        analyzer.visualize(data, names=names, sensor_type='thigh')
    
    def test_visualize_shank_data(self, sample_daphnet_data, mock_matplotlib):
        """Test visualizing shank data."""
        analyzer = DaphnetVisualizationAnalyzer()
        data, names = sample_daphnet_data
        
        # Should not raise an error
        analyzer.visualize(data, names=names, sensor_type='shank')
    
    def test_visualize_trunk_data(self, sample_daphnet_data, mock_matplotlib):
        """Test visualizing trunk data."""
        analyzer = DaphnetVisualizationAnalyzer()
        data, names = sample_daphnet_data
        
        # Should not raise an error
        analyzer.visualize(data, names=names, sensor_type='trunk')
    
    def test_visualize_unknown_sensor_type(self, sample_daphnet_data, mock_matplotlib):
        """Test visualizing with unknown sensor type."""
        analyzer = DaphnetVisualizationAnalyzer()
        data, names = sample_daphnet_data
        
        with patch('builtins.print') as mock_print:
            analyzer.visualize(data, names=names, sensor_type='unknown')
            mock_print.assert_called_with("Unknown sensor type: unknown")
    
    def test_visualize_dataset_index_out_of_range(self, sample_daphnet_data, mock_matplotlib):
        """Test visualizing with dataset index out of range."""
        analyzer = DaphnetVisualizationAnalyzer()
        data, names = sample_daphnet_data
        
        with patch('builtins.print') as mock_print:
            analyzer.visualize(data, names=names, dataset_index=10)
            mock_print.assert_called_with("Dataset index 10 out of range")
    
    def test_plot_thigh_data_empty(self, mock_matplotlib):
        """Test plotting thigh data with empty DataFrame."""
        analyzer = DaphnetVisualizationAnalyzer()
        
        # Create empty DataFrame
        df = pd.DataFrame()
        
        with patch('builtins.print') as mock_print:
            analyzer._plot_thigh_data(df, "test_dataset")
            mock_print.assert_called_with("No valid data to plot")
    
    def test_plot_shank_data_empty(self, mock_matplotlib):
        """Test plotting shank data with empty DataFrame."""
        analyzer = DaphnetVisualizationAnalyzer()
        
        # Create empty DataFrame
        df = pd.DataFrame()
        
        with patch('builtins.print') as mock_print:
            analyzer._plot_shank_data(df, "test_dataset")
            mock_print.assert_called_with("No valid data to plot")
    
    def test_plot_trunk_data_empty(self, mock_matplotlib):
        """Test plotting trunk data with empty DataFrame."""
        analyzer = DaphnetVisualizationAnalyzer()
        
        # Create empty DataFrame
        df = pd.DataFrame()
        
        with patch('builtins.print') as mock_print:
            analyzer._plot_trunk_data(df, "test_dataset")
            mock_print.assert_called_with("No valid data to plot")
    
    def test_plot_all_sensors_empty(self, mock_matplotlib):
        """Test plotting all sensors with empty DataFrame."""
        analyzer = DaphnetVisualizationAnalyzer()
        
        # Create empty DataFrame
        df = pd.DataFrame()
        
        with patch('builtins.print') as mock_print:
            analyzer._plot_all_sensors(df, "test_dataset")
            mock_print.assert_called_with("No valid data to plot")


class TestSensorStatisticsAnalyzer:
    """Test cases for SensorStatisticsAnalyzer."""
    
    def test_instantiation(self):
        """Test SensorStatisticsAnalyzer instantiation."""
        analyzer = SensorStatisticsAnalyzer()
        
        assert analyzer.name == "sensor_statistics"
        assert "statistical analysis" in analyzer.description.lower()
        assert analyzer.config['figsize'] == (20, 10)
        assert 'feature_markers' in analyzer.config
    
    def test_analyze_single_dataset(self, sample_daphnet_data):
        """Test analyzing a single dataset."""
        analyzer = SensorStatisticsAnalyzer()
        data, names = sample_daphnet_data
        
        result = analyzer.analyze(data[0])
        
        assert isinstance(result, dict)
        assert 'basic_stats' in result
        assert 'correlation_matrix' in result
        assert 'skewness' in result
        assert 'kurtosis' in result
        assert 'sensor_statistics' in result
    
    def test_analyze_multiple_datasets(self, sample_daphnet_data):
        """Test analyzing multiple datasets."""
        analyzer = SensorStatisticsAnalyzer()
        data, names = sample_daphnet_data
        
        result = analyzer.analyze(data)
        
        assert isinstance(result, dict)
        assert 'dataset_0' in result
        assert 'dataset_1' in result
        assert 'dataset_2' in result
    
    def test_analyze_empty_data(self):
        """Test analyzing empty data."""
        analyzer = SensorStatisticsAnalyzer()
        
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        result = analyzer.analyze(empty_df)
        
        assert isinstance(result, dict)
        assert 'basic_stats' in result
        assert 'correlation_matrix' in result
    
    def test_compute_statistics(self, sample_daphnet_data):
        """Test computing statistics for a dataset."""
        analyzer = SensorStatisticsAnalyzer()
        data, names = sample_daphnet_data
        
        result = analyzer._compute_statistics(data[0])
        
        assert isinstance(result, dict)
        assert 'basic_stats' in result
        assert 'correlation_matrix' in result
        assert 'skewness' in result
        assert 'kurtosis' in result
        assert 'sensor_statistics' in result
    
    def test_compute_statistics_single_column(self):
        """Test computing statistics for single column DataFrame."""
        analyzer = SensorStatisticsAnalyzer()
        
        # Create DataFrame with single column
        df = pd.DataFrame({'sensor1': [1, 2, 3, 4, 5]})
        
        result = analyzer._compute_statistics(df)
        
        assert isinstance(result, dict)
        assert 'basic_stats' in result
        assert 'correlation_matrix' in result  # Should be empty for single column
    
    def test_visualize_sensor_with_features(self, sample_sliding_windows, sample_features, mock_matplotlib):
        """Test visualizing sensor data with features."""
        analyzer = SensorStatisticsAnalyzer()
        
        # Should not raise an error
        analyzer.visualize(
            sample_sliding_windows, 
            sample_features,
            sensor_name='shank',
            start_idx=0,
            end_idx=100,
            num_windows=5
        )
    
    def test_visualize_sensor_not_found_in_windows(self, sample_features, mock_matplotlib):
        """Test visualizing with sensor not found in windows."""
        analyzer = SensorStatisticsAnalyzer()
        
        # Create windows without the requested sensor
        windows = [{'name': 'other_sensor', 'data': []}]
        
        with patch('builtins.print') as mock_print:
            analyzer.visualize(
                windows, 
                sample_features,
                sensor_name='shank'
            )
            mock_print.assert_called_with("Sensor 'shank' not found in sliding_windows.")
    
    def test_visualize_sensor_not_found_in_features(self, sample_sliding_windows, mock_matplotlib):
        """Test visualizing with sensor not found in features."""
        analyzer = SensorStatisticsAnalyzer()
        
        # Create features without the requested sensor
        features = [{'name': 'other_sensor', 'features': {}}]
        
        with patch('builtins.print') as mock_print:
            analyzer.visualize(
                sample_sliding_windows, 
                features,
                sensor_name='shank'
            )
            mock_print.assert_called_with("Sensor 'shank' not found in features.")
    
    def test_visualize_no_windows_in_range(self, sample_sliding_windows, sample_features, mock_matplotlib):
        """Test visualizing with no windows in specified range."""
        analyzer = SensorStatisticsAnalyzer()
        
        with patch('builtins.print') as mock_print:
            analyzer.visualize(
                sample_sliding_windows, 
                sample_features,
                sensor_name='shank',
                start_idx=1000,
                end_idx=2000
            )
            mock_print.assert_called_with("No windows found in the specified index range (1000 - 2000).")
    
    def test_plot_sensor_with_features_save(self, sample_sliding_windows, sample_features, mock_matplotlib):
        """Test plotting sensor with features and saving."""
        analyzer = SensorStatisticsAnalyzer()
        
        with patch('builtins.input', return_value='test_plot.png'):
            with patch('matplotlib.pyplot.savefig') as mock_savefig:
                analyzer.visualize(
                    sample_sliding_windows, 
                    sample_features,
                    sensor_name='shank',
                    save=True
                )
                mock_savefig.assert_called_once()


class TestEDAAnalyzerEdgeCases:
    """Test edge cases for EDA analyzers."""
    
    def test_analyze_data_with_nan_values(self):
        """Test analyzing data with NaN values."""
        analyzer = DaphnetVisualizationAnalyzer()
        
        # Create DataFrame with NaN values
        df = pd.DataFrame({
            'shank': [1, 2, np.nan, 4, 5],
            'thigh': [2, 3, 4, np.nan, 6],
            'trunk': [3, 4, 5, 6, 7],
            'annotations': [1, 2, 1, 2, 1]
        })
        
        result = analyzer.analyze(df)
        
        assert isinstance(result, dict)
        assert 'missing_values' in result
        assert result['missing_values']['shank'] == 1
        assert result['missing_values']['thigh'] == 1
    
    def test_analyze_data_with_inf_values(self):
        """Test analyzing data with infinite values."""
        analyzer = SensorStatisticsAnalyzer()
        
        # Create DataFrame with infinite values
        df = pd.DataFrame({
            'sensor1': [1, 2, np.inf, 4, 5],
            'sensor2': [2, 3, 4, 5, 6]
        })
        
        result = analyzer.analyze(df)
        
        assert isinstance(result, dict)
        assert 'basic_stats' in result
    
    def test_analyze_data_with_negative_values(self):
        """Test analyzing data with negative values."""
        analyzer = SensorStatisticsAnalyzer()
        
        # Create DataFrame with negative values
        df = pd.DataFrame({
            'sensor1': [-1, -2, 0, 2, 1],
            'sensor2': [2, 3, 4, 5, 6]
        })
        
        result = analyzer.analyze(df)
        
        assert isinstance(result, dict)
        assert 'basic_stats' in result
        assert 'sensor_statistics' in result
    
    def test_analyze_data_with_zero_variance(self):
        """Test analyzing data with zero variance."""
        analyzer = SensorStatisticsAnalyzer()
        
        # Create DataFrame with constant values (zero variance)
        df = pd.DataFrame({
            'sensor1': [5, 5, 5, 5, 5],
            'sensor2': [2, 3, 4, 5, 6]
        })
        
        result = analyzer.analyze(df)
        
        assert isinstance(result, dict)
        assert 'basic_stats' in result
        assert 'sensor_statistics' in result


class TestEDAAnalyzerPerformance:
    """Test performance aspects of EDA analyzers."""
    
    def test_analyze_large_dataset(self):
        """Test analyzing large dataset."""
        analyzer = SensorStatisticsAnalyzer()
        
        # Create large dataset
        large_data = pd.DataFrame({
            'sensor1': np.random.randn(10000),
            'sensor2': np.random.randn(10000),
            'sensor3': np.random.randn(10000)
        })
        
        import time
        start_time = time.time()
        
        result = analyzer.analyze(large_data)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete within reasonable time
        assert execution_time < 5.0  # 5 second threshold
        assert isinstance(result, dict)
    
    def test_visualize_large_dataset(self, mock_matplotlib):
        """Test visualizing large dataset."""
        analyzer = DaphnetVisualizationAnalyzer()
        
        # Create large dataset
        large_data = pd.DataFrame({
            'shank': np.random.randn(10000),
            'thigh': np.random.randn(10000),
            'trunk': np.random.randn(10000),
            'annotations': np.random.choice([1, 2], 10000)
        })
        
        # Should not raise an error
        analyzer.visualize(large_data, names=['large_dataset'], sensor_type='all')
    
    def test_memory_usage(self):
        """Test memory usage during analysis."""
        analyzer = SensorStatisticsAnalyzer()
        
        # Create moderately large dataset
        data = pd.DataFrame({
            'sensor1': np.random.randn(1000),
            'sensor2': np.random.randn(1000),
            'sensor3': np.random.randn(1000)
        })
        
        # Multiple analyses should not cause memory issues
        for _ in range(10):
            result = analyzer.analyze(data)
            assert isinstance(result, dict)
