import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from gaitsetpy.dataset.daphnet import DaphnetLoader

def test_daphnetloader_instantiation():
    loader = DaphnetLoader()
    assert loader.name == "daphnet"
    assert "sensors" in loader.metadata

def test_get_supported_formats():
    loader = DaphnetLoader()
    assert loader.get_supported_formats() == ['.txt']

def test_get_sensor_info():
    loader = DaphnetLoader()
    info = loader.get_sensor_info()
    assert 'sensors' in info
    assert 'components' in info
    assert 'sampling_frequency' in info

def test_get_annotation_info():
    loader = DaphnetLoader()
    ann = loader.get_annotation_info()
    assert 0 in ann and 1 in ann and 2 in ann

def test_create_sliding_windows_with_dummy_data():
    loader = DaphnetLoader()
    # Create dummy DataFrame
    df = pd.DataFrame({
        'shank_h_fd': np.arange(20),
        'shank_v': np.arange(20),
        'shank_h_l': np.arange(20),
        'thigh_h_fd': np.arange(20),
        'thigh_v': np.arange(20),
        'thigh_h_l': np.arange(20),
        'trunk_h_fd': np.arange(20),
        'trunk_v': np.arange(20),
        'trunk_h_l': np.arange(20),
        'annotations': np.ones(20)
    })
    df['time'] = np.arange(20)
    df = df.set_index('time')
    # Add magnitude columns as in loader
    df["thigh"] = np.sqrt(df["thigh_h_l"]**2 + df["thigh_v"]**2 + df["thigh_h_fd"]**2)
    df["shank"] = np.sqrt(df["shank_h_l"]**2 + df["shank_v"]**2 + df["shank_h_fd"]**2)
    df["trunk"] = np.sqrt(df["trunk_h_l"]**2 + df["trunk_v"]**2 + df["trunk_h_fd"]**2)
    # Reorder columns
    df = df[["shank", "shank_h_fd", "shank_v", "shank_h_l", 
            "thigh", "thigh_h_fd", "thigh_v", "thigh_h_l", 
            "trunk", "trunk_h_fd", "trunk_v", "trunk_h_l", "annotations"]]
    data = [df]
    names = ["dummy"]
    windows = loader.create_sliding_windows(data, names, window_size=5, step_size=2)
    assert isinstance(windows, list)
    assert windows[0]["name"] == "dummy"
    assert any(w["name"] == "annotations" for w in windows[0]["windows"])

@patch("gaitsetpy.dataset.daphnet.download_dataset")
@patch("gaitsetpy.dataset.daphnet.extract_dataset")
@patch("gaitsetpy.dataset.daphnet.glob")
def test_load_data_mocks(glob_mock, extract_mock, download_mock):
    loader = DaphnetLoader()
    # Mock glob to return a fake file list
    glob_mock.return_value = ["/tmp/S01.txt"]
    # Patch pandas.read_csv to return a dummy DataFrame
    with patch("pandas.read_csv") as read_csv_mock:
        df = pd.DataFrame({
            'time': np.arange(10),
            'shank_h_fd': np.arange(10),
            'shank_v': np.arange(10),
            'shank_h_l': np.arange(10),
            'thigh_h_fd': np.arange(10),
            'thigh_v': np.arange(10),
            'thigh_h_l': np.arange(10),
            'trunk_h_fd': np.arange(10),
            'trunk_v': np.arange(10),
            'trunk_h_l': np.arange(10),
            'annotations': np.ones(10)
        })
        read_csv_mock.return_value = df
        data, names = loader.load_data("/tmp")
        assert isinstance(data, list)
        assert isinstance(names, list)
        assert len(data) == 1
        assert len(names) == 1 