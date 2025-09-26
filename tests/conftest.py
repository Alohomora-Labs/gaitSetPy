"""
Pytest configuration and shared fixtures for GaitSetPy tests.

This module provides common fixtures, test data, and configuration
for all test modules in the GaitSetPy test suite.

Maintainer: @aharshit123456
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock
from typing import List, Dict, Any


@pytest.fixture
def sample_daphnet_data():
    """Create sample Daphnet dataset for testing."""
    np.random.seed(42)
    data = []
    names = []
    
    for i in range(3):  # 3 subjects
        df = pd.DataFrame({
            'time': np.arange(100),
            'shank_h_fd': np.random.randn(100),
            'shank_v': np.random.randn(100),
            'shank_h_l': np.random.randn(100),
            'thigh_h_fd': np.random.randn(100),
            'thigh_v': np.random.randn(100),
            'thigh_h_l': np.random.randn(100),
            'trunk_h_fd': np.random.randn(100),
            'trunk_v': np.random.randn(100),
            'trunk_h_l': np.random.randn(100),
            'annotations': np.random.choice([0, 1, 2], 100, p=[0.1, 0.7, 0.2])
        })
        df = df.set_index('time')
        
        # Calculate magnitude columns
        df["thigh"] = np.sqrt(df["thigh_h_l"]**2 + df["thigh_v"]**2 + df["thigh_h_fd"]**2)
        df["shank"] = np.sqrt(df["shank_h_l"]**2 + df["shank_v"]**2 + df["shank_h_fd"]**2)
        df["trunk"] = np.sqrt(df["trunk_h_l"]**2 + df["trunk_v"]**2 + df["trunk_h_fd"]**2)
        
        # Reorder columns
        df = df[["shank", "shank_h_fd", "shank_v", "shank_h_l", 
                "thigh", "thigh_h_fd", "thigh_v", "thigh_h_l", 
                "trunk", "trunk_h_fd", "trunk_v", "trunk_h_l", "annotations"]]
        
        data.append(df)
        names.append(f"S{i+1:02d}.txt")
    
    return data, names


@pytest.fixture
def sample_harup_data():
    """Create sample HAR-UP dataset for testing."""
    np.random.seed(42)
    data = []
    names = []
    
    for i in range(2):  # 2 subjects
        df = pd.DataFrame({
            'TIME': np.arange(100),
            'Belt_Acc_X': np.random.randn(100),
            'Belt_Acc_Y': np.random.randn(100),
            'Belt_Acc_Z': np.random.randn(100),
            'Belt_Gyro_X': np.random.randn(100),
            'Belt_Gyro_Y': np.random.randn(100),
            'Belt_Gyro_Z': np.random.randn(100),
            'activity_id': np.random.choice([1, 2, 3], 100),
            'subject_id': [i+1] * 100,
            'trial_id': [1] * 100,
            'activity_label': np.random.choice(['Walking', 'Sitting', 'Standing'], 100)
        })
        data.append(df)
        names.append(f"Subject_{i+1}_Activity_1_Trial_1.csv")
    
    return data, names


@pytest.fixture
def sample_physionet_data():
    """Create sample PhysioNet dataset for testing."""
    np.random.seed(42)
    data = []
    names = []
    
    for i in range(2):  # 2 subjects
        df = pd.DataFrame({
            'time': np.arange(100),
            'VGRF_L1': np.random.randn(100),
            'VGRF_L2': np.random.randn(100),
            'VGRF_L3': np.random.randn(100),
            'VGRF_L4': np.random.randn(100),
            'VGRF_L5': np.random.randn(100),
            'VGRF_L6': np.random.randn(100),
            'VGRF_L7': np.random.randn(100),
            'VGRF_L8': np.random.randn(100),
            'VGRF_R1': np.random.randn(100),
            'VGRF_R2': np.random.randn(100),
            'VGRF_R3': np.random.randn(100),
            'VGRF_R4': np.random.randn(100),
            'VGRF_R5': np.random.randn(100),
            'VGRF_R6': np.random.randn(100),
            'VGRF_R7': np.random.randn(100),
            'VGRF_R8': np.random.randn(100),
            'subject_type': ['Control' if i == 0 else 'Patient'] * 100,
            'label': ['Co' if i == 0 else 'Pt'] * 100
        })
        df = df.set_index('time')
        data.append(df)
        names.append(f"GaCo{i+1:02d}_01.txt" if i == 0 else f"GaPt{i+1:02d}_01.txt")
    
    return data, names


@pytest.fixture
def sample_sliding_windows():
    """Create sample sliding windows for testing."""
    np.random.seed(42)
    windows = []
    
    for sensor in ['shank', 'thigh', 'trunk']:
        window_data = []
        for i in range(5):  # 5 windows
            window_data.append(pd.Series(
                np.random.randn(10), 
                index=range(i*5, i*5+10),
                name=sensor
            ))
        windows.append({
            'name': sensor,
            'data': window_data
        })
    
    # Add annotations
    annotations = []
    for i in range(5):
        annotations.append(pd.Series(
            np.random.choice([1, 2], 10),
            index=range(i*5, i*5+10),
            name='annotations'
        ))
    windows.append({
        'name': 'annotations',
        'data': annotations
    })
    
    return windows


@pytest.fixture
def sample_features():
    """Create sample extracted features for testing."""
    np.random.seed(42)
    features = []
    
    for sensor in ['shank', 'thigh', 'trunk']:
        sensor_features = {
            'name': sensor,
            'features': {
                'mean': [np.random.randn() for _ in range(5)],
                'std': [np.random.randn() for _ in range(5)],
                'rms': [np.random.randn() for _ in range(5)],
                'dominant_frequency': [np.random.randn() for _ in range(5)],
                'entropy': [np.random.randn() for _ in range(5)]
            }
        }
        features.append(sensor_features)
    
    # Add annotations
    features.append({
        'name': 'annotations',
        'features': {},
        'annotations': [1, 2, 1, 2, 1]
    })
    
    return features


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_matplotlib():
    """Mock matplotlib to prevent GUI issues in tests."""
    with patch('matplotlib.pyplot.show') as mock_show:
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            yield mock_show, mock_savefig


@pytest.fixture
def mock_downloads():
    """Mock dataset downloads for testing."""
    with patch('gaitsetpy.dataset.utils.download_dataset') as mock_download:
        with patch('gaitsetpy.dataset.utils.extract_dataset') as mock_extract:
            mock_download.return_value = None
            mock_extract.return_value = None
            yield mock_download, mock_extract


class TestDataGenerator:
    """Utility class for generating test data."""
    
    @staticmethod
    def create_sine_wave(length: int, frequency: float, sampling_rate: float = 100) -> np.ndarray:
        """Create a sine wave for testing."""
        t = np.linspace(0, length/sampling_rate, length)
        return np.sin(2 * np.pi * frequency * t)
    
    @staticmethod
    def create_noisy_signal(length: int, noise_level: float = 0.1) -> np.ndarray:
        """Create a noisy signal for testing."""
        signal = TestDataGenerator.create_sine_wave(length, 1.0)
        noise = np.random.randn(length) * noise_level
        return signal + noise
    
    @staticmethod
    def create_step_signal(length: int, step_position: int = None) -> np.ndarray:
        """Create a step signal for testing."""
        if step_position is None:
            step_position = length // 2
        signal = np.zeros(length)
        signal[step_position:] = 1
        return signal


@pytest.fixture
def test_data_generator():
    """Provide TestDataGenerator instance."""
    return TestDataGenerator


# Test markers for different test categories
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "requires_data: Tests that require actual dataset files")
    config.addinivalue_line("markers", "requires_gpu: Tests that require GPU")
    config.addinivalue_line("markers", "visualization: Tests that create visualizations")


# Test collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add unit marker to tests in unit directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Add integration marker to tests in integration directory
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add slow marker to tests with 'slow' in name
        if "slow" in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Add visualization marker to tests with 'plot' or 'visualize' in name
        if any(keyword in item.name.lower() for keyword in ['plot', 'visualize', 'show']):
            item.add_marker(pytest.mark.visualization)
