"""
dataset: Handles loading and processing of supported datasets.

Supported datasets:
- Daphnet
- MobiFall
- Arduous
"""

from .daphnet import load_daphnet_data, create_sliding_windows
from .mobifall import load_mobifall_data
from .arduous import load_arduous_data
from .utils import download_dataset, extract_dataset, sliding_window