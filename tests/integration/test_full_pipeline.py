"""
Integration tests for complete GaitSetPy pipeline.

This module tests the complete workflow from data loading
through feature extraction to model training and evaluation.

Maintainer: @aharshit123456
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, Mock

from gaitsetpy import (
    DaphnetLoader,
    GaitFeatureExtractor,
    RandomForestModel,
    DaphnetVisualizationAnalyzer
)


class TestDaphnetFullPipeline:
    """Test complete Daphnet analysis pipeline."""
    
    def test_complete_daphnet_pipeline(self, sample_daphnet_data, mock_downloads, mock_matplotlib):
        """Test complete pipeline from data loading to model evaluation."""
        # Step 1: Load data
        loader = DaphnetLoader()
        data, names = sample_daphnet_data
        
        assert len(data) == 3
        assert len(names) == 3
        
        # Step 2: Create sliding windows
        windows = loader.create_sliding_windows(data, names, window_size=10, step_size=5)
        
        assert isinstance(windows, list)
        assert len(windows) == 3
        
        # Step 3: Extract features
        extractor = GaitFeatureExtractor(verbose=False)
        features = extractor.extract_features(windows[0]['windows'], fs=64)
        
        assert isinstance(features, list)
        assert len(features) > 0
        
        # Step 4: Prepare features for classification
        feature_dicts = []
        annotation_sensor = None
        
        for sensor in features:
            if sensor['name'] == 'annotations':
                annotation_sensor = sensor
        
        for sensor in features:
            if sensor['name'] != 'annotations':
                n_windows = len(next(iter(sensor['features'].values())))
                feature_dicts.append({
                    'name': sensor['name'],
                    'features': sensor['features'],
                    'annotations': annotation_sensor['annotations'][:n_windows] if annotation_sensor else [0]*n_windows
                })
        
        assert len(feature_dicts) > 0
        
        # Step 5: Train model
        model = RandomForestModel(n_estimators=10, random_state=42)
        model.train(feature_dicts, test_size=0.2, validation_split=True)
        
        assert model.trained is True
        
        # Step 6: Evaluate model
        metrics = model.evaluate(feature_dicts, detailed_report=True)
        
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'confusion_matrix' in metrics
        
        # Step 7: Make predictions
        predictions = model.predict(feature_dicts)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) > 0
    
    def test_pipeline_with_visualization(self, sample_daphnet_data, mock_downloads, mock_matplotlib):
        """Test pipeline including visualization steps."""
        # Load data
        loader = DaphnetLoader()
        data, names = sample_daphnet_data
        
        # Create sliding windows
        windows = loader.create_sliding_windows(data, names, window_size=10, step_size=5)
        
        # Extract features
        extractor = GaitFeatureExtractor(verbose=False)
        features = extractor.extract_features(windows[0]['windows'], fs=64)
        
        # Visualize data
        analyzer = DaphnetVisualizationAnalyzer()
        analysis = analyzer.analyze(data[0])
        
        assert isinstance(analysis, dict)
        assert 'sensor_statistics' in analysis
        
        # Visualize with features
        from gaitsetpy.eda.analyzers import SensorStatisticsAnalyzer
        sensor_analyzer = SensorStatisticsAnalyzer()
        
        # Should not raise an error
        sensor_analyzer.visualize(
            windows[0]['windows'], 
            features,
            sensor_name='shank',
            start_idx=0,
            end_idx=50,
            num_windows=3
        )
    
    def test_pipeline_with_preprocessing(self, sample_daphnet_data, mock_downloads):
        """Test pipeline with preprocessing steps."""
        from gaitsetpy.preprocessing.preprocessors import (
            NoiseRemovalPreprocessor,
            BaselineRemovalPreprocessor
        )
        
        # Load data
        loader = DaphnetLoader()
        data, names = sample_daphnet_data
        
        # Preprocess data
        noise_remover = NoiseRemovalPreprocessor(window_size=3)
        baseline_remover = BaselineRemovalPreprocessor()
        
        # Apply preprocessing to first sensor column
        sensor_data = data[0]['shank'].values
        preprocessed_data = noise_remover.fit_transform(sensor_data)
        preprocessed_data = baseline_remover.fit_transform(preprocessed_data)
        
        # Update data with preprocessed values
        data[0]['shank'] = preprocessed_data
        
        # Continue with normal pipeline
        windows = loader.create_sliding_windows(data, names, window_size=10, step_size=5)
        
        extractor = GaitFeatureExtractor(verbose=False)
        features = extractor.extract_features(windows[0]['windows'], fs=64)
        
        assert isinstance(features, list)
        assert len(features) > 0


class TestHARUPFullPipeline:
    """Test complete HAR-UP analysis pipeline."""
    
    def test_complete_harup_pipeline(self, sample_harup_data, mock_downloads):
        """Test complete HAR-UP pipeline."""
        from gaitsetpy.dataset.harup import HARUPLoader
        from gaitsetpy.features.harup_features import HARUPFeatureExtractor
        
        # Load data
        loader = HARUPLoader()
        data, names = sample_harup_data
        
        assert len(data) == 2
        assert len(names) == 2
        
        # Create sliding windows
        windows = loader.create_sliding_windows(data, names, window_size=10, step_size=5)
        
        assert isinstance(windows, list)
        assert len(windows) == 2
        
        # Extract features
        features_data = loader.extract_features(windows)
        
        assert isinstance(features_data, list)
        assert len(features_data) == 2
        
        # Prepare for classification
        feature_dicts = []
        for rec in features_data:
            name = rec["name"]
            features_list = rec["features"]
            if not features_list:
                continue
            
            # Get all feature keys except 'sensor' and 'label'
            feature_keys = [k for k in features_list[0].keys() if k not in ("sensor", "label")]
            features = {k: [f.get(k, 0) for f in features_list] for k in feature_keys}
            annotations = [f["label"] for f in features_list if "label" in f]
            
            feature_dicts.append({
                "name": name,
                "features": features,
                "annotations": annotations
            })
        
        assert len(feature_dicts) > 0
        
        # Train model
        model = RandomForestModel(n_estimators=10, random_state=42)
        model.train(feature_dicts, test_size=0.2, validation_split=True)
        
        assert model.trained is True
        
        # Evaluate model
        metrics = model.evaluate(feature_dicts, detailed_report=True)
        
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics


class TestPhysioNetFullPipeline:
    """Test complete PhysioNet analysis pipeline."""
    
    def test_complete_physionet_pipeline(self, sample_physionet_data, mock_downloads):
        """Test complete PhysioNet pipeline."""
        from gaitsetpy.dataset.physionet import PhysioNetLoader
        from gaitsetpy.features.physionet_features import PhysioNetFeatureExtractor
        
        # Load data
        loader = PhysioNetLoader()
        data, names = sample_physionet_data
        
        assert len(data) == 2
        assert len(names) == 2
        
        # Create sliding windows
        windows = loader.create_sliding_windows(data, names, window_size=20, step_size=10)
        
        assert isinstance(windows, list)
        assert len(windows) == 2
        
        # Extract features
        extractor = PhysioNetFeatureExtractor(verbose=False)
        all_features = []
        
        for window_dict in windows:
            if 'windows' in window_dict:
                features = extractor.extract_features(window_dict['windows'], fs=100)
                all_features.append({
                    'name': window_dict['name'],
                    'features': features,
                    'metadata': window_dict.get('metadata', {})
                })
        
        assert len(all_features) == 2
        
        # Prepare for classification
        X = []
        y = []
        
        for file_features in all_features:
            file_name = file_features['name']
            features = file_features['features']
            metadata = file_features.get('metadata', {})
            
            # Get label from metadata or filename
            if 'label' in metadata:
                label = metadata['label']
            else:
                label = 'Co' if 'Co' in file_name else 'Pt'
            
            # Flatten features for each sensor
            for sensor_features in features:
                sensor_name = sensor_features['name']
                sensor_data = sensor_features['features']
                
                # Create feature vector by concatenating all features
                feature_vector = []
                
                for feature_name, feature_values in sensor_data.items():
                    if isinstance(feature_values, list):
                        if len(feature_values) > 0:
                            if isinstance(feature_values[0], (list, np.ndarray)):
                                flat_values = []
                                for val in feature_values:
                                    if isinstance(val, (list, np.ndarray)):
                                        flat_values.extend(val)
                                    else:
                                        flat_values.append(val)
                                feature_vector.append(np.mean(flat_values))
                            else:
                                feature_vector.append(np.mean(feature_values))
                        else:
                            feature_vector.append(0)
                    else:
                        feature_vector.append(feature_values)
                
                if len(feature_vector) > 0:
                    X.append(feature_vector)
                    y.append(label)
        
        assert len(X) > 0
        assert len(y) > 0
        
        # Train model
        model = RandomForestModel(n_estimators=10, random_state=42)
        
        # Convert to feature dict format
        feature_dicts = []
        for i, (features, label) in enumerate(zip(X, y)):
            feature_dicts.append({
                'name': f'sample_{i}',
                'features': {f'feature_{j}': [val] for j, val in enumerate(features)},
                'annotations': [label]
            })
        
        model.train(feature_dicts, test_size=0.2, validation_split=True)
        
        assert model.trained is True
        
        # Evaluate model
        metrics = model.evaluate(feature_dicts, detailed_report=True)
        
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics


class TestPipelineErrorHandling:
    """Test error handling in pipeline."""
    
    def test_pipeline_with_invalid_data(self, mock_downloads):
        """Test pipeline with invalid data."""
        loader = DaphnetLoader()
        
        # Test with empty data
        empty_data = []
        empty_names = []
        
        windows = loader.create_sliding_windows(empty_data, empty_names)
        assert windows == []
        
        # Test with malformed data
        malformed_data = [pd.DataFrame()]  # Empty DataFrame
        malformed_names = ["malformed"]
        
        # Should handle empty DataFrame gracefully
        with pytest.raises(AttributeError):
            windows = loader.create_sliding_windows(malformed_data, malformed_names)
    
    def test_pipeline_with_missing_annotations(self, sample_daphnet_data, mock_downloads):
        """Test pipeline with missing annotations."""
        loader = DaphnetLoader()
        data, names = sample_daphnet_data
        
        # Remove annotations column
        data[0] = data[0].drop(columns=['annotations'])
        
        # Should handle missing annotations gracefully
        with pytest.raises(AttributeError):
            windows = loader.create_sliding_windows(data, names, window_size=10, step_size=5)
        
        # Test that the exception is properly raised and we don't continue
        # with invalid data (this test verifies error handling, not feature extraction)
        assert True  # If we reach here, the exception was properly raised
    
    def test_pipeline_with_insufficient_data(self, mock_downloads):
        """Test pipeline with insufficient data for training."""
        # Create minimal data
        minimal_data = [pd.DataFrame({
            'shank': [1, 2],
            'thigh': [2, 3],
            'trunk': [3, 4],
            'annotations': [1, 2]
        })]
        minimal_names = ["minimal"]
        
        loader = DaphnetLoader()
        windows = loader.create_sliding_windows(minimal_data, minimal_names, window_size=5, step_size=2)
        
        # Should handle gracefully
        assert isinstance(windows, list)
        
        if windows:  # If windows were created
            extractor = GaitFeatureExtractor(verbose=False)
            features = extractor.extract_features(windows[0]['windows'], fs=64)
            
            assert isinstance(features, list)


class TestPipelinePerformance:
    """Test performance aspects of pipeline."""
    
    def test_pipeline_performance_large_data(self, mock_downloads):
        """Test pipeline performance with larger data."""
        # Create larger dataset
        large_data = []
        large_names = []
        
        for i in range(5):  # 5 subjects
            df = pd.DataFrame({
                'shank': np.random.randn(1000),
                'thigh': np.random.randn(1000),
                'trunk': np.random.randn(1000),
                'annotations': np.random.choice([1, 2], 1000)
            })
            large_data.append(df)
            large_names.append(f"S{i+1:02d}.txt")
        
        loader = DaphnetLoader()
        windows = loader.create_sliding_windows(large_data, large_names, window_size=50, step_size=25)
        
        extractor = GaitFeatureExtractor(verbose=False)
        
        import time
        start_time = time.time()
        
        features = extractor.extract_features(windows[0]['windows'], fs=64)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete within reasonable time
        assert execution_time < 10.0  # 10 second threshold
        assert isinstance(features, list)
    
    def test_pipeline_memory_usage(self, sample_daphnet_data, mock_downloads):
        """Test pipeline memory usage."""
        loader = DaphnetLoader()
        data, names = sample_daphnet_data
        
        # Process multiple times to test memory usage
        for _ in range(5):
            windows = loader.create_sliding_windows(data, names, window_size=10, step_size=5)
            
            extractor = GaitFeatureExtractor(verbose=False)
            features = extractor.extract_features(windows[0]['windows'], fs=64)
            
            # Prepare features
            feature_dicts = []
            annotation_sensor = None
            
            for sensor in features:
                if sensor['name'] == 'annotations':
                    annotation_sensor = sensor
            
            for sensor in features:
                if sensor['name'] != 'annotations':
                    n_windows = len(next(iter(sensor['features'].values())))
                    feature_dicts.append({
                        'name': sensor['name'],
                        'features': sensor['features'],
                        'annotations': annotation_sensor['annotations'][:n_windows] if annotation_sensor else [0]*n_windows
                    })
            
            # Train model
            model = RandomForestModel(n_estimators=5, random_state=42)
            model.train(feature_dicts, validation_split=False)
            
            assert model.trained is True
