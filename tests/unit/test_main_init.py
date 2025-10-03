"""
Unit tests for the main __init__.py module in GaitSetPy.

This module tests the main package initialization, utility functions,
and workflow functions that are exported from the main package.

Maintainer: @aharshit123456
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import patch, Mock, MagicMock
from typing import List, Dict, Any

import gaitsetpy
from gaitsetpy import (
    get_all_managers,
    get_system_info,
    load_and_analyze_daphnet,
    load_and_analyze_physionet,
    train_gait_classifier,
    __version__,
    __author__
)


class TestMainInit:
    """Test cases for main package initialization."""
    
    def test_version_and_author(self):
        """Test that version and author are properly defined."""
        assert hasattr(gaitsetpy, '__version__')
        assert hasattr(gaitsetpy, '__author__')
        assert isinstance(__version__, str)
        assert isinstance(__author__, str)
        assert len(__version__) > 0
        assert len(__author__) > 0
    
    def test_core_imports(self):
        """Test that core components are properly imported."""
        from gaitsetpy import (
            BaseDatasetLoader,
            BaseFeatureExtractor,
            BasePreprocessor,
            BaseEDAAnalyzer,
            BaseClassificationModel,
            DatasetManager,
            FeatureManager,
            PreprocessingManager,
            EDAManager,
            ClassificationManager
        )
        
        # Test that classes are imported
        assert BaseDatasetLoader is not None
        assert BaseFeatureExtractor is not None
        assert BasePreprocessor is not None
        assert BaseEDAAnalyzer is not None
        assert BaseClassificationModel is not None
        assert DatasetManager is not None
        assert FeatureManager is not None
        assert PreprocessingManager is not None
        assert EDAManager is not None
        assert ClassificationManager is not None
    
    def test_dataset_imports(self):
        """Test that dataset loaders are properly imported."""
        from gaitsetpy import (
            DaphnetLoader,
            MobiFallLoader,
            ArduousLoader,
            PhysioNetLoader,
            HARUPLoader,
            get_dataset_manager,
            get_available_datasets,
            load_dataset
        )
        
        assert DaphnetLoader is not None
        assert MobiFallLoader is not None
        assert ArduousLoader is not None
        assert PhysioNetLoader is not None
        assert HARUPLoader is not None
        assert get_dataset_manager is not None
        assert get_available_datasets is not None
        assert load_dataset is not None
    
    def test_feature_imports(self):
        """Test that feature extractors are properly imported."""
        from gaitsetpy import (
            GaitFeatureExtractor,
            LBPFeatureExtractor,
            FourierSeriesFeatureExtractor,
            PhysioNetFeatureExtractor,
            get_feature_manager,
            get_available_extractors,
            extract_features
        )
        
        assert GaitFeatureExtractor is not None
        assert LBPFeatureExtractor is not None
        assert FourierSeriesFeatureExtractor is not None
        assert PhysioNetFeatureExtractor is not None
        assert get_feature_manager is not None
        assert get_available_extractors is not None
        assert extract_features is not None
    
    def test_preprocessing_imports(self):
        """Test that preprocessors are properly imported."""
        from gaitsetpy import (
            ClippingPreprocessor,
            NoiseRemovalPreprocessor,
            OutlierRemovalPreprocessor,
            BaselineRemovalPreprocessor,
            DriftRemovalPreprocessor,
            HighFrequencyNoiseRemovalPreprocessor,
            LowFrequencyNoiseRemovalPreprocessor,
            ArtifactRemovalPreprocessor,
            TrendRemovalPreprocessor,
            DCOffsetRemovalPreprocessor,
            get_preprocessing_manager,
            get_available_preprocessors,
            preprocess_data,
            create_preprocessing_pipeline
        )
        
        # Test that all preprocessors are imported
        preprocessors = [
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
        ]
        
        for preprocessor in preprocessors:
            assert preprocessor is not None
        
        # Test utility functions
        assert get_preprocessing_manager is not None
        assert get_available_preprocessors is not None
        assert preprocess_data is not None
        assert create_preprocessing_pipeline is not None
    
    def test_eda_imports(self):
        """Test that EDA analyzers are properly imported."""
        from gaitsetpy import (
            DaphnetVisualizationAnalyzer,
            SensorStatisticsAnalyzer,
            get_eda_manager,
            get_available_analyzers,
            analyze_data,
            visualize_data,
            plot_daphnet_data,
            analyze_sensor_statistics,
            plot_sensor_features
        )
        
        assert DaphnetVisualizationAnalyzer is not None
        assert SensorStatisticsAnalyzer is not None
        assert get_eda_manager is not None
        assert get_available_analyzers is not None
        assert analyze_data is not None
        assert visualize_data is not None
        assert plot_daphnet_data is not None
        assert analyze_sensor_statistics is not None
        assert plot_sensor_features is not None
    
    def test_classification_imports(self):
        """Test that classification models are properly imported."""
        from gaitsetpy import (
            RandomForestModel,
            get_classification_manager,
            get_available_models,
            train_model,
            predict,
            evaluate_model_performance,
            create_random_forest,
            train_random_forest
        )
        
        assert RandomForestModel is not None
        assert get_classification_manager is not None
        assert get_available_models is not None
        assert train_model is not None
        assert predict is not None
        assert evaluate_model_performance is not None
        assert create_random_forest is not None
        assert train_random_forest is not None


class TestUtilityFunctions:
    """Test cases for utility functions in main __init__.py."""
    
    def test_get_all_managers(self):
        """Test get_all_managers function."""
        managers = get_all_managers()
        
        assert isinstance(managers, dict)
        assert 'dataset' in managers
        assert 'feature' in managers
        assert 'preprocessing' in managers
        assert 'eda' in managers
        assert 'classification' in managers
        
        # Test that all managers are instances of their respective classes
        from gaitsetpy import (
            DatasetManager,
            FeatureManager,
            PreprocessingManager,
            EDAManager,
            ClassificationManager
        )
        
        assert isinstance(managers['dataset'], DatasetManager)
        assert isinstance(managers['feature'], FeatureManager)
        assert isinstance(managers['preprocessing'], PreprocessingManager)
        assert isinstance(managers['eda'], EDAManager)
        assert isinstance(managers['classification'], ClassificationManager)
    
    def test_get_system_info(self):
        """Test get_system_info function."""
        info = get_system_info()
        
        assert isinstance(info, dict)
        assert 'version' in info
        assert 'author' in info
        assert 'available_datasets' in info
        assert 'available_extractors' in info
        assert 'available_preprocessors' in info
        assert 'available_analyzers' in info
        assert 'available_models' in info
        assert 'architecture' in info
        
        # Test specific values
        assert info['version'] == __version__
        assert info['author'] == __author__
        assert info['architecture'] == 'Modular with singleton design pattern'
        
        # Test that available components are lists
        assert isinstance(info['available_datasets'], list)
        assert isinstance(info['available_extractors'], list)
        assert isinstance(info['available_preprocessors'], list)
        assert isinstance(info['available_analyzers'], list)
        assert isinstance(info['available_models'], list)


class TestWorkflowFunctions:
    """Test cases for workflow functions in main __init__.py."""
    
    @patch('gaitsetpy.DaphnetLoader')
    @patch('gaitsetpy.GaitFeatureExtractor')
    @patch('gaitsetpy.DaphnetVisualizationAnalyzer')
    def test_load_and_analyze_daphnet(self, mock_analyzer, mock_extractor, mock_loader):
        """Test load_and_analyze_daphnet workflow function."""
        # Mock the loader
        mock_loader_instance = Mock()
        mock_loader.return_value = mock_loader_instance
        
        # Mock data
        mock_data = [{'sensor1': [1, 2, 3], 'sensor2': [4, 5, 6]}]
        mock_names = ['dataset1']
        mock_windows = [{'name': 'dataset1', 'windows': [np.array([1, 2, 3])]}]
        
        mock_loader_instance.load_data.return_value = (mock_data, mock_names)
        mock_loader_instance.create_sliding_windows.return_value = mock_windows
        
        # Mock extractor
        mock_extractor_instance = Mock()
        mock_extractor.return_value = mock_extractor_instance
        mock_extractor_instance.extract_features.return_value = {'mean': [1.0], 'std': [0.5]}
        
        # Mock analyzer
        mock_analyzer_instance = Mock()
        mock_analyzer.return_value = mock_analyzer_instance
        mock_analyzer_instance.analyze.return_value = {'analysis': 'results'}
        
        # Test the workflow
        result = load_and_analyze_daphnet('/test/data', sensor_type='all', window_size=192)
        
        assert isinstance(result, dict)
        assert 'data' in result
        assert 'names' in result
        assert 'windows' in result
        assert 'features' in result
        assert 'analysis' in result
        assert 'loader' in result
        assert 'extractor' in result
        assert 'analyzer' in result
        
        # Verify method calls
        mock_loader_instance.load_data.assert_called_once_with('/test/data')
        mock_loader_instance.create_sliding_windows.assert_called_once_with(
            mock_data, mock_names, window_size=192
        )
        mock_extractor_instance.extract_features.assert_called_once()
        mock_analyzer_instance.analyze.assert_called_once_with(mock_data)
    
    @patch('gaitsetpy.PhysioNetLoader')
    @patch('gaitsetpy.PhysioNetFeatureExtractor')
    def test_load_and_analyze_physionet(self, mock_extractor, mock_loader):
        """Test load_and_analyze_physionet workflow function."""
        # Mock the loader
        mock_loader_instance = Mock()
        mock_loader.return_value = mock_loader_instance
        
        # Mock data
        mock_data = [{'sensor1': [1, 2, 3], 'sensor2': [4, 5, 6]}]
        mock_names = ['dataset1']
        mock_windows = [{'name': 'dataset1', 'windows': [np.array([1, 2, 3])], 'metadata': {}}]
        
        mock_loader_instance.load_data.return_value = (mock_data, mock_names)
        mock_loader_instance.create_sliding_windows.return_value = mock_windows
        mock_loader_instance.get_labels.return_value = [0, 1, 0]
        
        # Mock extractor
        mock_extractor_instance = Mock()
        mock_extractor.return_value = mock_extractor_instance
        mock_extractor_instance.extract_features.return_value = {'mean': [1.0], 'std': [0.5]}
        
        # Test the workflow
        result = load_and_analyze_physionet('/test/data', window_size=600, step_size=100)
        
        assert isinstance(result, dict)
        assert 'data' in result
        assert 'names' in result
        assert 'windows' in result
        assert 'features' in result
        assert 'labels' in result
        assert 'loader' in result
        assert 'extractor' in result
        
        # Verify method calls
        mock_loader_instance.load_data.assert_called_once_with('/test/data')
        mock_loader_instance.create_sliding_windows.assert_called_once_with(
            mock_data, mock_names, window_size=600, step_size=100
        )
        mock_loader_instance.get_labels.assert_called_once()
        mock_extractor_instance.extract_features.assert_called()
    
    @patch('gaitsetpy.RandomForestModel')
    def test_train_gait_classifier_random_forest(self, mock_rf_model):
        """Test train_gait_classifier with Random Forest."""
        # Mock the model
        mock_model_instance = Mock()
        mock_rf_model.return_value = mock_model_instance
        
        # Mock features
        features = [
            {'name': 'sensor1', 'features': {'mean': [1.0, 2.0]}, 'annotations': [0, 1]}
        ]
        
        # Test the workflow
        result = train_gait_classifier(features, model_type='random_forest', n_estimators=100)
        
        # Verify model creation and training
        mock_rf_model.assert_called_once_with(n_estimators=100)
        mock_model_instance.train.assert_called_once_with(features, n_estimators=100)
        assert result == mock_model_instance
    
    def test_train_gait_classifier_unsupported_model(self):
        """Test train_gait_classifier with unsupported model type."""
        features = [
            {'name': 'sensor1', 'features': {'mean': [1.0, 2.0]}, 'annotations': [0, 1]}
        ]
        
        with pytest.raises(ValueError, match="Model type 'unsupported' not supported"):
            train_gait_classifier(features, model_type='unsupported')


class TestLegacyImports:
    """Test cases for legacy function imports."""
    
    def test_legacy_dataset_imports(self):
        """Test that legacy dataset functions are properly imported."""
        from gaitsetpy import (
            load_daphnet_data,
            create_sliding_windows,
            load_mobifall_data,
            load_arduous_data,
            load_physionet_data,
            create_physionet_windows,
            load_harup_data,
            create_harup_windows,
            extract_harup_features,
            download_dataset,
            extract_dataset,
            sliding_window
        )
        
        # Test that all functions are imported
        legacy_functions = [
            load_daphnet_data,
            create_sliding_windows,
            load_mobifall_data,
            load_arduous_data,
            load_physionet_data,
            create_physionet_windows,
            load_harup_data,
            create_harup_windows,
            extract_harup_features,
            download_dataset,
            extract_dataset,
            sliding_window
        ]
        
        for func in legacy_functions:
            assert func is not None
            assert callable(func)
    
    def test_legacy_feature_imports(self):
        """Test that legacy feature functions are properly imported."""
        from gaitsetpy import (
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
            get_mean_for_windows,
            get_standard_deviation_for_windows,
            get_variance_for_windows
        )
        
        # Test that all functions are imported and callable
        legacy_functions = [
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
            get_mean_for_windows,
            get_standard_deviation_for_windows,
            get_variance_for_windows
        ]
        
        for func in legacy_functions:
            assert func is not None
            assert callable(func)
    
    def test_legacy_preprocessing_imports(self):
        """Test that legacy preprocessing functions are properly imported."""
        from gaitsetpy import (
            clip_sliding_windows,
            remove_noise,
            remove_outliers,
            remove_baseline,
            remove_drift,
            remove_artifacts,
            remove_trend,
            remove_dc_offset,
            remove_high_frequency_noise,
            remove_low_frequency_noise
        )
        
        # Test that all functions are imported and callable
        legacy_functions = [
            clip_sliding_windows,
            remove_noise,
            remove_outliers,
            remove_baseline,
            remove_drift,
            remove_artifacts,
            remove_trend,
            remove_dc_offset,
            remove_high_frequency_noise,
            remove_low_frequency_noise
        ]
        
        for func in legacy_functions:
            assert func is not None
            assert callable(func)
    
    def test_legacy_eda_imports(self):
        """Test that legacy EDA functions are properly imported."""
        from gaitsetpy import (
            plot_thigh_data,
            plot_shank_data,
            plot_trunk_data,
            plot_all_data,
            plot_all_thigh_data,
            plot_all_shank_data,
            plot_all_trunk_data,
            plot_all_datasets,
            plot_sensor_with_features
        )
        
        # Test that all functions are imported and callable
        legacy_functions = [
            plot_thigh_data,
            plot_shank_data,
            plot_trunk_data,
            plot_all_data,
            plot_all_thigh_data,
            plot_all_shank_data,
            plot_all_trunk_data,
            plot_all_datasets,
            plot_sensor_with_features
        ]
        
        for func in legacy_functions:
            assert func is not None
            assert callable(func)
    
    def test_legacy_classification_imports(self):
        """Test that legacy classification functions are properly imported."""
        from gaitsetpy import (
            create_random_forest_model,
            preprocess_features,
            evaluate_model
        )
        
        # Test that all functions are imported and callable
        legacy_functions = [
            create_random_forest_model,
            preprocess_features,
            evaluate_model
        ]
        
        for func in legacy_functions:
            assert func is not None
            assert callable(func)


class TestAllExports:
    """Test that all expected exports are available."""
    
    def test_all_exports_available(self):
        """Test that all items in __all__ are available."""
        from gaitsetpy import __all__
        
        # Test that __all__ is defined
        assert hasattr(gaitsetpy, '__all__')
        assert isinstance(__all__, list)
        assert len(__all__) > 0
        
        # Test that all items in __all__ are actually available
        for item_name in __all__:
            assert hasattr(gaitsetpy, item_name), f"Item {item_name} not found in gaitsetpy"
            item = getattr(gaitsetpy, item_name)
            assert item is not None, f"Item {item_name} is None"
    
    def test_expected_exports_present(self):
        """Test that expected key exports are present in __all__."""
        from gaitsetpy import __all__
        
        # Core architecture
        expected_core = [
            'BaseDatasetLoader',
            'BaseFeatureExtractor',
            'BasePreprocessor',
            'BaseEDAAnalyzer',
            'BaseClassificationModel',
            'DatasetManager',
            'FeatureManager',
            'PreprocessingManager',
            'EDAManager',
            'ClassificationManager'
        ]
        
        for item in expected_core:
            assert item in __all__, f"Core item {item} not in __all__"
        
        # Utility functions
        expected_utils = [
            'get_all_managers',
            'get_system_info',
            'get_available_datasets',
            'get_available_extractors',
            'get_available_preprocessors',
            'get_available_analyzers',
            'get_available_models'
        ]
        
        for item in expected_utils:
            assert item in __all__, f"Utility item {item} not in __all__"
        
        # Workflow functions
        expected_workflows = [
            'load_and_analyze_daphnet',
            'load_and_analyze_physionet',
            'train_gait_classifier'
        ]
        
        for item in expected_workflows:
            assert item in __all__, f"Workflow item {item} not in __all__"
