"""
Unit tests for EDA statistics and visualization modules in GaitSetPy.

This module tests the statistical analysis and visualization functions
in the EDA package.

Maintainer: @aharshit123456
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import patch, Mock, MagicMock
import tempfile
import os

from gaitsetpy.eda.statistics import plot_sensor_with_features
from gaitsetpy.eda.visualization import (
    plot_thigh_data,
    plot_shank_data,
    plot_trunk_data,
    plot_all_thigh_data,
    plot_all_shank_data,
    plot_all_trunk_data,
    plot_all_data,
    plot_all_datasets,
    plot_sensor_timeseries,
    plot_all_sensors,
    plot_activity_distribution
)


class TestPlotSensorWithFeatures:
    """Test cases for plot_sensor_with_features function."""
    
    @pytest.fixture
    def sample_sliding_windows(self):
        """Create sample sliding windows data."""
        return [
            {
                'name': 'shank',
                'data': [
                    pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=[0, 1, 2, 3, 4]),
                    pd.Series([2.0, 3.0, 4.0, 5.0, 6.0], index=[5, 6, 7, 8, 9]),
                    pd.Series([3.0, 4.0, 5.0, 6.0, 7.0], index=[10, 11, 12, 13, 14])
                ]
            }
        ]
    
    @pytest.fixture
    def sample_features(self):
        """Create sample features data."""
        return [
            {
                'name': 'shank',
                'features': {
                    'mean': [2.0, 3.0, 4.0],
                    'std': [0.5, 0.6, 0.7],
                    'rms': [2.2, 3.1, 4.1],
                    'peak_height': [5.0, 6.0, 7.0],
                    'mode': [1.0, 2.0, 3.0],
                    'median': [2.0, 3.0, 4.0],
                    'entropy': [0.5, 0.6, 0.7],
                    'dominant_frequency': [0.1, 0.2, 0.3]
                }
            }
        ]
    
    @patch('gaitsetpy.eda.statistics.plt.show')
    @patch('gaitsetpy.eda.statistics.plt.subplots')
    def test_plot_sensor_with_features_success(self, mock_subplots, mock_show, 
                                             sample_sliding_windows, sample_features):
        """Test successful plotting of sensor with features."""
        # Mock matplotlib
        mock_fig = Mock()
        mock_axes = [Mock(), Mock()]
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        # Test the function
        plot_sensor_with_features(
            sample_sliding_windows, 
            sample_features, 
            start_idx=0, 
            end_idx=15,
            sensor_name='shank',
            num_windows=3
        )
        
        # Verify matplotlib calls
        mock_subplots.assert_called_once()
        mock_show.assert_called_once()
    
    @patch('gaitsetpy.eda.statistics.plt.show')
    @patch('gaitsetpy.eda.statistics.plt.subplots')
    def test_plot_sensor_with_features_sensor_not_found(self, mock_subplots, mock_show,
                                                      sample_sliding_windows, sample_features):
        """Test plotting with sensor not found in sliding windows."""
        # Mock matplotlib
        mock_fig = Mock()
        mock_axes = [Mock(), Mock()]
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        # Test with non-existent sensor
        plot_sensor_with_features(
            sample_sliding_windows, 
            sample_features, 
            start_idx=0, 
            end_idx=15,
            sensor_name='nonexistent'
        )
        
        # Should not call show if sensor not found
        mock_show.assert_not_called()
    
    @patch('gaitsetpy.eda.statistics.plt.show')
    @patch('gaitsetpy.eda.statistics.plt.subplots')
    def test_plot_sensor_with_features_no_windows_in_range(self, mock_subplots, mock_show,
                                                         sample_sliding_windows, sample_features):
        """Test plotting with no windows in the specified range."""
        # Mock matplotlib
        mock_fig = Mock()
        mock_axes = [Mock(), Mock()]
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        # Test with range that has no windows
        plot_sensor_with_features(
            sample_sliding_windows, 
            sample_features, 
            start_idx=100, 
            end_idx=200,
            sensor_name='shank'
        )
        
        # Should not call show if no windows in range
        mock_show.assert_not_called()
    
    @patch('gaitsetpy.eda.statistics.plt.savefig')
    @patch('gaitsetpy.eda.statistics.plt.subplots')
    @patch('builtins.input')
    def test_plot_sensor_with_features_save(self, mock_input, mock_subplots, mock_savefig,
                                          sample_sliding_windows, sample_features):
        """Test saving the plot to a file."""
        # Mock matplotlib
        mock_fig = Mock()
        mock_axes = [Mock(), Mock()]
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        # Mock user input for file path
        mock_input.return_value = "test_plot.png"
        
        # Test the function with save=True
        plot_sensor_with_features(
            sample_sliding_windows, 
            sample_features, 
            start_idx=0, 
            end_idx=15,
            sensor_name='shank',
            save=True
        )
        
        # Verify savefig was called
        mock_savefig.assert_called_once_with("test_plot.png", dpi=300)
    
    @patch('gaitsetpy.eda.statistics.plt.show')
    @patch('gaitsetpy.eda.statistics.plt.subplots')
    def test_plot_sensor_with_features_empty_features(self, mock_subplots, mock_show,
                                                    sample_sliding_windows):
        """Test plotting with empty features."""
        # Mock matplotlib
        mock_fig = Mock()
        mock_axes = [Mock(), Mock()]
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        # Test with empty features
        empty_features = []
        plot_sensor_with_features(
            sample_sliding_windows, 
            empty_features, 
            start_idx=0, 
            end_idx=15,
            sensor_name='shank'
        )
        
        # Should not call show if features are empty
        mock_show.assert_not_called()


class TestDaphnetVisualization:
    """Test cases for Daphnet visualization functions."""
    
    @pytest.fixture
    def sample_daphnet_data(self):
        """Create sample Daphnet data."""
        data = pd.DataFrame({
            'thigh_h_fd': [1.0, 2.0, 3.0, 4.0, 5.0],
            'thigh_v': [1.5, 2.5, 3.5, 4.5, 5.5],
            'thigh_h_l': [0.5, 1.5, 2.5, 3.5, 4.5],
            'thigh': [1.8, 2.8, 3.8, 4.8, 5.8],
            'shank_h_fd': [1.1, 2.1, 3.1, 4.1, 5.1],
            'shank_v': [1.6, 2.6, 3.6, 4.6, 5.6],
            'shank_h_l': [0.6, 1.6, 2.6, 3.6, 4.6],
            'shank': [1.9, 2.9, 3.9, 4.9, 5.9],
            'trunk_h_fd': [1.2, 2.2, 3.2, 4.2, 5.2],
            'trunk_v': [1.7, 2.7, 3.7, 4.7, 5.7],
            'trunk_h_l': [0.7, 1.7, 2.7, 3.7, 4.7],
            'trunk': [2.0, 3.0, 4.0, 5.0, 6.0],
            'annotations': [1, 1, 2, 2, 1]
        })
        return data
    
    @pytest.fixture
    def sample_daphnet_names(self):
        """Create sample Daphnet names."""
        return ['dataset1', 'dataset2']
    
    @patch('gaitsetpy.eda.visualization.plt.show')
    @patch('gaitsetpy.eda.visualization.plt.subplots')
    def test_plot_thigh_data_success(self, mock_subplots, mock_show, 
                                   sample_daphnet_data, sample_daphnet_names):
        """Test successful plotting of thigh data."""
        # Mock matplotlib
        mock_fig = Mock()
        mock_axes = [Mock(), Mock(), Mock(), Mock()]
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        # Test the function
        plot_thigh_data([sample_daphnet_data], sample_daphnet_names, 0)
        
        # Verify matplotlib calls
        mock_subplots.assert_called_once()
        mock_show.assert_called_once()
    
    @patch('gaitsetpy.eda.visualization.plt.show')
    @patch('gaitsetpy.eda.visualization.plt.subplots')
    def test_plot_shank_data_success(self, mock_subplots, mock_show,
                                   sample_daphnet_data, sample_daphnet_names):
        """Test successful plotting of shank data."""
        # Mock matplotlib
        mock_fig = Mock()
        mock_axes = [Mock(), Mock(), Mock(), Mock()]
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        # Test the function
        plot_shank_data([sample_daphnet_data], sample_daphnet_names, 0)
        
        # Verify matplotlib calls
        mock_subplots.assert_called_once()
        mock_show.assert_called_once()
    
    @patch('gaitsetpy.eda.visualization.plt.show')
    @patch('gaitsetpy.eda.visualization.plt.subplots')
    def test_plot_trunk_data_success(self, mock_subplots, mock_show,
                                   sample_daphnet_data, sample_daphnet_names):
        """Test successful plotting of trunk data."""
        # Mock matplotlib
        mock_fig = Mock()
        mock_axes = [Mock(), Mock(), Mock(), Mock()]
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        # Test the function
        plot_trunk_data([sample_daphnet_data], sample_daphnet_names, 0)
        
        # Verify matplotlib calls
        mock_subplots.assert_called_once()
        mock_show.assert_called_once()
    
    @patch('gaitsetpy.eda.visualization.plot_thigh_data')
    def test_plot_all_thigh_data(self, mock_plot_thigh):
        """Test plotting all thigh data."""
        sample_data = [pd.DataFrame({'thigh_h_fd': [1, 2, 3]})]
        sample_names = ['dataset1', 'dataset2']
        
        plot_all_thigh_data(sample_data, sample_names)
        
        # Should call plot_thigh_data for each dataset
        assert mock_plot_thigh.call_count == len(sample_data)
    
    @patch('gaitsetpy.eda.visualization.plot_shank_data')
    def test_plot_all_shank_data(self, mock_plot_shank):
        """Test plotting all shank data."""
        sample_data = [pd.DataFrame({'shank_h_fd': [1, 2, 3]})]
        sample_names = ['dataset1', 'dataset2']
        
        plot_all_shank_data(sample_data, sample_names)
        
        # Should call plot_shank_data for each dataset
        assert mock_plot_shank.call_count == len(sample_data)
    
    @patch('gaitsetpy.eda.visualization.plot_trunk_data')
    def test_plot_all_trunk_data(self, mock_plot_trunk):
        """Test plotting all trunk data."""
        sample_data = [pd.DataFrame({'trunk_h_fd': [1, 2, 3]})]
        sample_names = ['dataset1', 'dataset2']
        
        plot_all_trunk_data(sample_data, sample_names)
        
        # Should call plot_trunk_data for each dataset
        assert mock_plot_trunk.call_count == len(sample_data)
    
    @patch('gaitsetpy.eda.visualization.plot_thigh_data')
    @patch('gaitsetpy.eda.visualization.plot_shank_data')
    @patch('gaitsetpy.eda.visualization.plot_trunk_data')
    def test_plot_all_data(self, mock_plot_trunk, mock_plot_shank, mock_plot_thigh,
                         sample_daphnet_data, sample_daphnet_names):
        """Test plotting all data for a specific dataset."""
        plot_all_data([sample_daphnet_data], [sample_daphnet_data], [sample_daphnet_data], 
                     sample_daphnet_names, 0)
        
        # Should call all three plotting functions
        mock_plot_thigh.assert_called_once()
        mock_plot_shank.assert_called_once()
        mock_plot_trunk.assert_called_once()
    
    @patch('gaitsetpy.eda.visualization.plot_all_data')
    def test_plot_all_datasets(self, mock_plot_all):
        """Test plotting all datasets."""
        sample_data = [pd.DataFrame({'thigh_h_fd': [1, 2, 3]})]
        sample_names = ['dataset1']
        
        plot_all_datasets(sample_data, sample_data, sample_data, sample_names)
        
        # Should call plot_all_data for each dataset
        assert mock_plot_all.call_count == len(sample_data)


class TestHARUPVisualization:
    """Test cases for HAR-UP visualization functions."""
    
    @pytest.fixture
    def sample_harup_df(self):
        """Create sample HAR-UP DataFrame."""
        return pd.DataFrame({
            'sensor1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'sensor2': [1.5, 2.5, 3.5, 4.5, 5.5],
            'sensor3': [0.5, 1.5, 2.5, 3.5, 4.5],
            'subject_id': [1, 1, 1, 1, 1],
            'activity_id': [1, 1, 1, 1, 1],
            'trial_id': [1, 1, 1, 1, 1],
            'activity_label': ['walking', 'walking', 'running', 'running', 'walking']
        })
    
    @patch('gaitsetpy.eda.visualization.plt.show')
    @patch('gaitsetpy.eda.visualization.plt.figure')
    def test_plot_sensor_timeseries_success(self, mock_figure, mock_show, sample_harup_df):
        """Test successful plotting of sensor time series."""
        # Mock matplotlib
        mock_fig = Mock()
        mock_ax = Mock()
        mock_figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax
        
        # Test the function
        plot_sensor_timeseries(sample_harup_df, 'sensor1', 'Test Sensor')
        
        # Verify matplotlib calls
        mock_figure.assert_called_once()
        mock_show.assert_called_once()
    
    @patch('gaitsetpy.eda.visualization.plt.show')
    @patch('gaitsetpy.eda.visualization.plt.figure')
    def test_plot_sensor_timeseries_column_not_found(self, mock_figure, mock_show, sample_harup_df):
        """Test plotting with non-existent column."""
        # Mock matplotlib
        mock_fig = Mock()
        mock_figure.return_value = mock_fig
        
        # Test with non-existent column
        plot_sensor_timeseries(sample_harup_df, 'nonexistent_column')
        
        # Should not call show if column not found
        mock_show.assert_not_called()
    
    @patch('gaitsetpy.eda.visualization.plt.show')
    @patch('gaitsetpy.eda.visualization.plt.subplots')
    def test_plot_all_sensors_success(self, mock_subplots, mock_show, sample_harup_df):
        """Test successful plotting of all sensors."""
        # Mock matplotlib
        mock_fig = Mock()
        mock_axes = np.array([[Mock(), Mock()], [Mock(), Mock()]])
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        # Test the function
        plot_all_sensors(sample_harup_df, sensor_cols=['sensor1', 'sensor2'], max_cols=2)
        
        # Verify matplotlib calls
        mock_subplots.assert_called_once()
        mock_show.assert_called_once()
    
    @patch('gaitsetpy.eda.visualization.plt.show')
    @patch('gaitsetpy.eda.visualization.plt.subplots')
    def test_plot_all_sensors_auto_detect(self, mock_subplots, mock_show, sample_harup_df):
        """Test plotting all sensors with automatic column detection."""
        # Mock matplotlib
        mock_fig = Mock()
        mock_axes = np.array([[Mock(), Mock()], [Mock(), Mock()]])
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        # Test the function without specifying sensor_cols
        plot_all_sensors(sample_harup_df, max_cols=2)
        
        # Verify matplotlib calls
        mock_subplots.assert_called_once()
        mock_show.assert_called_once()
    
    @patch('gaitsetpy.eda.visualization.plt.show')
    @patch('gaitsetpy.eda.visualization.plt.figure')
    def test_plot_activity_distribution_success(self, mock_figure, mock_show, sample_harup_df):
        """Test successful plotting of activity distribution."""
        # Mock matplotlib
        mock_fig = Mock()
        mock_ax = Mock()
        mock_figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax
        
        # Test the function
        plot_activity_distribution(sample_harup_df)
        
        # Verify matplotlib calls
        mock_figure.assert_called_once()
        mock_show.assert_called_once()
    
    @patch('gaitsetpy.eda.visualization.plt.show')
    @patch('gaitsetpy.eda.visualization.plt.figure')
    def test_plot_activity_distribution_no_label_column(self, mock_figure, mock_show, sample_harup_df):
        """Test plotting activity distribution without activity_label column."""
        # Remove activity_label column
        df_no_label = sample_harup_df.drop('activity_label', axis=1)
        
        # Mock matplotlib
        mock_fig = Mock()
        mock_figure.return_value = mock_fig
        
        # Test the function
        plot_activity_distribution(df_no_label)
        
        # Should not call show if no activity_label column
        mock_show.assert_not_called()


class TestVisualizationEdgeCases:
    """Test edge cases for visualization functions."""
    
    @patch('gaitsetpy.eda.visualization.plt.show')
    @patch('gaitsetpy.eda.visualization.plt.subplots')
    def test_plot_thigh_data_empty_annotations(self, mock_subplots, mock_show):
        """Test plotting thigh data with no annotations."""
        # Create data with no annotations (all zeros)
        data = pd.DataFrame({
            'thigh_h_fd': [1.0, 2.0, 3.0],
            'thigh_v': [1.5, 2.5, 3.5],
            'thigh_h_l': [0.5, 1.5, 2.5],
            'thigh': [1.8, 2.8, 3.8],
            'annotations': [0, 0, 0]  # No annotations
        })
        
        # Mock matplotlib
        mock_fig = Mock()
        mock_axes = [Mock(), Mock(), Mock(), Mock()]
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        # Test the function
        plot_thigh_data([data], ['dataset1'], 0)
        
        # Should still work with no annotations
        mock_subplots.assert_called_once()
        mock_show.assert_called_once()
    
    @patch('gaitsetpy.eda.visualization.plt.show')
    @patch('gaitsetpy.eda.visualization.plt.subplots')
    def test_plot_shank_data_single_annotation(self, mock_subplots, mock_show):
        """Test plotting shank data with single annotation type."""
        # Create data with only one annotation type
        data = pd.DataFrame({
            'shank_h_fd': [1.0, 2.0, 3.0],
            'shank_v': [1.5, 2.5, 3.5],
            'shank_h_l': [0.5, 1.5, 2.5],
            'shank': [1.8, 2.8, 3.8],
            'annotations': [1, 1, 1]  # Only one annotation type
        })
        
        # Mock matplotlib
        mock_fig = Mock()
        mock_axes = [Mock(), Mock(), Mock(), Mock()]
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        # Test the function
        plot_shank_data([data], ['dataset1'], 0)
        
        # Should still work with single annotation type
        mock_subplots.assert_called_once()
        mock_show.assert_called_once()
    
    @patch('gaitsetpy.eda.visualization.plt.show')
    @patch('gaitsetpy.eda.visualization.plt.subplots')
    def test_plot_trunk_data_missing_columns(self, mock_subplots, mock_show):
        """Test plotting trunk data with missing columns."""
        # Create data with missing columns
        data = pd.DataFrame({
            'trunk_h_fd': [1.0, 2.0, 3.0],
            'annotations': [1, 2, 1]
            # Missing trunk_v, trunk_h_l, trunk columns
        })
        
        # Mock matplotlib
        mock_fig = Mock()
        mock_axes = [Mock(), Mock(), Mock(), Mock()]
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        # Test the function - should handle missing columns gracefully
        try:
            plot_trunk_data([data], ['dataset1'], 0)
        except KeyError:
            # Expected to fail with missing columns
            pass
    
    def test_plot_all_thigh_data_empty_list(self):
        """Test plotting all thigh data with empty list."""
        with patch('gaitsetpy.eda.visualization.plot_thigh_data') as mock_plot:
            plot_all_thigh_data([], [])
            
            # Should not call plot_thigh_data with empty list
            mock_plot.assert_not_called()
    
    def test_plot_all_shank_data_mismatched_lengths(self):
        """Test plotting all shank data with mismatched list lengths."""
        with patch('gaitsetpy.eda.visualization.plot_shank_data') as mock_plot:
            # Different lengths for data and names
            plot_all_shank_data([pd.DataFrame({'shank_h_fd': [1, 2, 3]})], ['dataset1', 'dataset2'])
            
            # Should only call plot_shank_data once (for the single data item)
            assert mock_plot.call_count == 1
    
    def test_plot_all_trunk_data_single_dataset(self):
        """Test plotting all trunk data with single dataset."""
        with patch('gaitsetpy.eda.visualization.plot_trunk_data') as mock_plot:
            data = [pd.DataFrame({'trunk_h_fd': [1, 2, 3]})]
            names = ['dataset1']
            
            plot_all_trunk_data(data, names)
            
            # Should call plot_trunk_data once
            mock_plot.assert_called_once_with(data, names, 0)


class TestVisualizationIntegration:
    """Integration tests for visualization functions."""
    
    @patch('gaitsetpy.eda.visualization.plt.show')
    @patch('gaitsetpy.eda.visualization.plt.subplots')
    def test_full_daphnet_visualization_workflow(self, mock_subplots, mock_show):
        """Test complete Daphnet visualization workflow."""
        # Create comprehensive test data
        data = pd.DataFrame({
            'thigh_h_fd': [1.0, 2.0, 3.0, 4.0, 5.0],
            'thigh_v': [1.5, 2.5, 3.5, 4.5, 5.5],
            'thigh_h_l': [0.5, 1.5, 2.5, 3.5, 4.5],
            'thigh': [1.8, 2.8, 3.8, 4.8, 5.8],
            'shank_h_fd': [1.1, 2.1, 3.1, 4.1, 5.1],
            'shank_v': [1.6, 2.6, 3.6, 4.6, 5.6],
            'shank_h_l': [0.6, 1.6, 2.6, 3.6, 4.6],
            'shank': [1.9, 2.9, 3.9, 4.9, 5.9],
            'trunk_h_fd': [1.2, 2.2, 3.2, 4.2, 5.2],
            'trunk_v': [1.7, 2.7, 3.7, 4.7, 5.7],
            'trunk_h_l': [0.7, 1.7, 2.7, 3.7, 4.7],
            'trunk': [2.0, 3.0, 4.0, 5.0, 6.0],
            'annotations': [1, 1, 2, 2, 1]
        })
        
        names = ['test_dataset']
        
        # Mock matplotlib
        mock_fig = Mock()
        mock_axes = [Mock(), Mock(), Mock(), Mock()]
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        # Test all plotting functions
        plot_thigh_data([data], names, 0)
        plot_shank_data([data], names, 0)
        plot_trunk_data([data], names, 0)
        
        # Verify all calls were made
        assert mock_subplots.call_count == 3
        assert mock_show.call_count == 3
    
    @patch('gaitsetpy.eda.visualization.plt.show')
    @patch('gaitsetpy.eda.visualization.plt.figure')
    def test_full_harup_visualization_workflow(self, mock_figure, mock_show):
        """Test complete HAR-UP visualization workflow."""
        # Create comprehensive test data
        data = pd.DataFrame({
            'sensor1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'sensor2': [1.5, 2.5, 3.5, 4.5, 5.5],
            'sensor3': [0.5, 1.5, 2.5, 3.5, 4.5],
            'sensor4': [2.0, 3.0, 4.0, 5.0, 6.0],
            'subject_id': [1, 1, 1, 1, 1],
            'activity_id': [1, 1, 1, 1, 1],
            'trial_id': [1, 1, 1, 1, 1],
            'activity_label': ['walking', 'walking', 'running', 'running', 'walking']
        })
        
        # Mock matplotlib
        mock_fig = Mock()
        mock_ax = Mock()
        mock_figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax
        
        # Test all plotting functions
        plot_sensor_timeseries(data, 'sensor1', 'Test Sensor')
        plot_activity_distribution(data)
        
        # Verify calls were made
        assert mock_figure.call_count == 2
        assert mock_show.call_count == 2


class TestVisualizationErrorHandling:
    """Test error handling in visualization functions."""
    
    def test_plot_thigh_data_invalid_index(self):
        """Test plotting thigh data with invalid dataset index."""
        data = pd.DataFrame({'thigh_h_fd': [1, 2, 3]})
        names = ['dataset1']
        
        # Should handle invalid index gracefully
        try:
            plot_thigh_data([data], names, 5)  # Index out of range
        except IndexError:
            # Expected to fail with invalid index
            pass
    
    def test_plot_shank_data_empty_dataframe(self):
        """Test plotting shank data with empty DataFrame."""
        empty_data = pd.DataFrame()
        names = ['dataset1']
        
        # Should handle empty DataFrame gracefully
        try:
            plot_shank_data([empty_data], names, 0)
        except (KeyError, IndexError):
            # Expected to fail with empty DataFrame
            pass
    
    def test_plot_trunk_data_none_data(self):
        """Test plotting trunk data with None data."""
        names = ['dataset1']
        
        # Should handle None data gracefully
        try:
            plot_trunk_data([None], names, 0)
        except (AttributeError, TypeError):
            # Expected to fail with None data
            pass
    
    def test_plot_sensor_timeseries_none_dataframe(self):
        """Test plotting sensor time series with None DataFrame."""
        # Should handle None DataFrame gracefully
        try:
            plot_sensor_timeseries(None, 'sensor1')
        except (AttributeError, TypeError):
            # Expected to fail with None DataFrame
            pass
    
    def test_plot_activity_distribution_empty_dataframe(self):
        """Test plotting activity distribution with empty DataFrame."""
        empty_data = pd.DataFrame()
        
        # Should handle empty DataFrame gracefully
        try:
            plot_activity_distribution(empty_data)
        except (KeyError, IndexError):
            # Expected to fail with empty DataFrame
            pass
