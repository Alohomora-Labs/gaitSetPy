"""
dataset: Handles loading and processing of supported datasets.

This module provides both the new class-based dataset loaders and legacy function-based API.
All dataset loaders inherit from BaseDatasetLoader and are registered with the DatasetManager.

Supported datasets:
- Daphnet: Freezing of Gait dataset
- MobiFall: Fall detection dataset
- Arduous: Daily activity recognition dataset
- PhysioNet: VGRF dataset for Parkinson's disease gait analysis
- HAR-UP: Multimodal System for Fall Detection and Human Activity Recognition

"""

# Import the new class-based loaders
from .daphnet import DaphnetLoader
from .mobifall import MobiFallLoader
from .arduous import ArduousLoader
from .physionet import PhysioNetLoader
from .harup import HARUPLoader

# Import legacy functions for backward compatibility
from .daphnet import load_daphnet_data, create_sliding_windows
from .mobifall import load_mobifall_data
from .arduous import load_arduous_data
from .physionet import load_physionet_data, create_physionet_windows
from .harup import load_harup_data, create_harup_windows, extract_harup_features
from .utils import download_dataset, extract_dataset, sliding_window

# Import managers
from ..core.managers import DatasetManager

# Register all dataset loaders with the manager
def _register_datasets():
    """Register all available dataset loaders with the DatasetManager."""
    manager = DatasetManager()
    manager.register_dataset("daphnet", DaphnetLoader)
    manager.register_dataset("mobifall", MobiFallLoader)
    manager.register_dataset("arduous", ArduousLoader)
    manager.register_dataset("physionet", PhysioNetLoader)
    manager.register_dataset("harup", HARUPLoader)

# Auto-register datasets when module is imported
_register_datasets()

# Convenient access to the dataset manager
def get_dataset_manager():
    """Get the singleton DatasetManager instance."""
    return DatasetManager()

# Helper function to get available datasets
def get_available_datasets():
    """Get list of available dataset names."""
    return DatasetManager().get_available_components()

# Helper function to load dataset using manager
def load_dataset(name: str, data_dir: str, **kwargs):
    """
    Load a dataset using the DatasetManager.
    
    Args:
        name: Name of the dataset loader
        data_dir: Directory containing the dataset
        **kwargs: Additional arguments for the loader
        
    Returns:
        Dataset loader instance with loaded data
    """
    return DatasetManager().load_dataset(name, data_dir, **kwargs)

__all__ = [
    # New class-based loaders
    'DaphnetLoader',
    'MobiFallLoader', 
    'ArduousLoader',
    'PhysioNetLoader',
    'HARUPLoader',
    # Legacy functions for backward compatibility
    'load_daphnet_data',
    'create_sliding_windows',
    'load_mobifall_data',
    'load_arduous_data',
    'load_physionet_data',
    'create_physionet_windows',
    'load_harup_data',
    'create_harup_windows',
    'extract_harup_features',
    'download_dataset',
    'extract_dataset',
    'sliding_window',
    # Manager functions
    'get_dataset_manager',
    'get_available_datasets',
    'load_dataset'
]