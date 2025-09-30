"""
Tests for UrFall dataset loader.
"""

import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch, MagicMock
from gaitsetpy.dataset.urfall import UrFallLoader


def test_urfallloader_instantiation():
    """Test that UrFallLoader can be instantiated."""
    loader = UrFallLoader()
    assert loader.name == "urfall"
    assert "data_types" in loader.metadata
    assert "feature_columns" in loader.metadata


def test_get_supported_formats():
    """Test that supported formats are correctly returned."""
    loader = UrFallLoader()
    formats = loader.get_supported_formats()
    assert '.csv' in formats
    assert '.zip' in formats
    assert '.mp4' in formats


def test_get_sensor_info():
    """Test that sensor information is correctly returned."""
    loader = UrFallLoader()
    info = loader.get_sensor_info()
    assert 'data_types' in info
    assert 'camera' in info
    assert 'sampling_frequency' in info
    assert info['camera'] == 'cam0'


def test_get_activity_info():
    """Test that activity information is correctly returned."""
    loader = UrFallLoader()
    activities = loader.get_activity_info()
    assert -1 in activities
    assert 0 in activities
    assert 1 in activities
    assert activities[1] == 'Lying on ground'


def test_get_feature_info():
    """Test that feature information is correctly returned."""
    loader = UrFallLoader()
    features = loader.get_feature_info()
    assert 'HeightWidthRatio' in features
    assert 'MajorMinorRatio' in features
    assert 'P40' in features


def test_create_sliding_windows_with_dummy_data():
    """Test sliding window creation with dummy feature data."""
    loader = UrFallLoader()
    
    # Create dummy feature DataFrame
    df = pd.DataFrame({
        'sequence_name': ['fall-01'] * 100,
        'frame_number': list(range(100)),
        'label': [1] * 100,
        'HeightWidthRatio': np.random.rand(100),
        'MajorMinorRatio': np.random.rand(100),
        'BoundingBoxOccupancy': np.random.rand(100),
        'activity_id': [1] * 100,
        'activity_type': ['fall'] * 100
    })
    
    data = [df]
    names = ["urfall-cam0-falls"]
    
    windows = loader.create_sliding_windows(data, names, window_size=30, step_size=15)
    
    assert isinstance(windows, list)
    assert len(windows) > 0
    assert windows[0]["name"] == "urfall-cam0-falls"
    assert "windows" in windows[0]
    
    # Check that features are windowed
    window_names = [w["name"] for w in windows[0]["windows"]]
    assert "HeightWidthRatio" in window_names
    assert "labels" in window_names


def test_load_features_with_mock_files(tmp_path):
    """Test loading features from mock CSV files."""
    loader = UrFallLoader()
    
    # Create mock CSV files
    falls_data = {
        'col0': ['fall-01', 'fall-01', 'fall-02'],
        'col1': [1, 2, 1],
        'col2': [1, 1, -1],
        'col3': [1.5, 1.6, 1.4],
        'col4': [2.0, 2.1, 1.9],
        'col5': [0.8, 0.75, 0.85],
        'col6': [50, 55, 45],
        'col7': [0.9, 0.85, 0.95],
        'col8': [1700, 1650, 1750],
        'col9': [100, 120, 90],
        'col10': [0.6, 0.65, 0.7]
    }
    falls_df = pd.DataFrame(falls_data)
    falls_csv = tmp_path / "urfall-cam0-falls.csv"
    falls_df.to_csv(falls_csv, index=False, header=False)
    
    adls_data = {
        'col0': ['adl-01', 'adl-01', 'adl-02'],
        'col1': [1, 2, 1],
        'col2': [-1, -1, -1],
        'col3': [1.2, 1.3, 1.1],
        'col4': [1.8, 1.9, 1.7],
        'col5': [0.7, 0.72, 0.68],
        'col6': [40, 42, 38],
        'col7': [0.8, 0.82, 0.78],
        'col8': [1600, 1620, 1580],
        'col9': [80, 85, 75],
        'col10': [0.5, 0.52, 0.48]
    }
    adls_df = pd.DataFrame(adls_data)
    adls_csv = tmp_path / "urfall-cam0-adls.csv"
    adls_df.to_csv(adls_csv, index=False, header=False)
    
    # Load the data
    data, names = loader.load_data(str(tmp_path), data_types=['features'])
    
    assert len(data) == 2
    assert len(names) == 2
    assert "urfall-cam0-falls" in names
    assert "urfall-cam0-adls" in names
    
    # Check that data has correct columns
    falls_loaded = data[names.index("urfall-cam0-falls")]
    assert 'sequence_name' in falls_loaded.columns
    assert 'label' in falls_loaded.columns
    assert 'HeightWidthRatio' in falls_loaded.columns
    assert 'activity_id' in falls_loaded.columns
    assert all(falls_loaded['activity_id'] == 1)


def test_load_features_only_falls(tmp_path):
    """Test loading only fall sequences."""
    loader = UrFallLoader()
    
    # Create mock falls CSV file
    falls_data = {
        'col0': ['fall-01'] * 5,
        'col1': list(range(5)),
        'col2': [1] * 5,
        'col3': [1.5] * 5,
        'col4': [2.0] * 5,
        'col5': [0.8] * 5,
        'col6': [50] * 5,
        'col7': [0.9] * 5,
        'col8': [1700] * 5,
        'col9': [100] * 5,
        'col10': [0.6] * 5
    }
    falls_df = pd.DataFrame(falls_data)
    falls_csv = tmp_path / "urfall-cam0-falls.csv"
    falls_df.to_csv(falls_csv, index=False, header=False)
    
    # Load only falls
    data, names = loader.load_data(str(tmp_path), data_types=['features'], 
                                   use_falls=True, use_adls=False)
    
    assert len(data) == 1
    assert names[0] == "urfall-cam0-falls"


def test_get_file_paths_for_video():
    """Test getting file paths for video files."""
    loader = UrFallLoader()
    
    # Should work but return empty dict if files don't exist
    paths = loader.get_file_paths("/nonexistent", "video")
    assert isinstance(paths, dict)
    assert len(paths) == 0  # No files exist


def test_metadata_structure():
    """Test that metadata has all required fields."""
    loader = UrFallLoader()
    
    assert 'data_types' in loader.metadata
    assert 'fall_sequences' in loader.metadata
    assert 'adl_sequences' in loader.metadata
    assert 'feature_columns' in loader.metadata
    assert 'feature_descriptions' in loader.metadata
    
    # Check sequences
    assert len(loader.metadata['fall_sequences']) == 30
    assert len(loader.metadata['adl_sequences']) == 20


def test_invalid_data_type():
    """Test that invalid data type raises an error."""
    loader = UrFallLoader()
    
    with pytest.raises(ValueError):
        loader.load_data("/tmp", data_types=['invalid_type'])


# Test legacy function wrappers
def test_legacy_load_urfall_data(tmp_path):
    """Test legacy load_urfall_data function."""
    from gaitsetpy.dataset.urfall import load_urfall_data
    
    # Create mock CSV file
    falls_data = {
        'col0': ['fall-01'] * 5,
        'col1': list(range(5)),
        'col2': [1] * 5,
        'col3': [1.5] * 5,
        'col4': [2.0] * 5,
        'col5': [0.8] * 5,
        'col6': [50] * 5,
        'col7': [0.9] * 5,
        'col8': [1700] * 5,
        'col9': [100] * 5,
        'col10': [0.6] * 5
    }
    falls_df = pd.DataFrame(falls_data)
    falls_csv = tmp_path / "urfall-cam0-falls.csv"
    falls_df.to_csv(falls_csv, index=False, header=False)
    
    data, names = load_urfall_data(str(tmp_path), use_falls=True, use_adls=False)
    assert len(data) > 0


def test_legacy_create_urfall_windows():
    """Test legacy create_urfall_windows function."""
    from gaitsetpy.dataset.urfall import create_urfall_windows
    
    # Create dummy data
    df = pd.DataFrame({
        'sequence_name': ['fall-01'] * 50,
        'frame_number': list(range(50)),
        'label': [1] * 50,
        'HeightWidthRatio': np.random.rand(50),
        'activity_id': [1] * 50
    })
    
    windows = create_urfall_windows([df], ["test"], window_size=20, step_size=10)
    assert isinstance(windows, list)
    assert len(windows) > 0
