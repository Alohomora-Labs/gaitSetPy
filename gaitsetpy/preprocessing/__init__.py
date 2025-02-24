"""
preprocessing: Preprocessing pipelines for gait data.

Features:
TODO: 
- Segmentation
- Normalization
- Custom preprocessing adapters

Maintainer: @aharshit123456
"""

# from .pipeline import preprocess, segment_data

from .pipeline import clip_sliding_windows, remove_noise, remove_outliers, remove_baseline, remove_drift, remove_artifacts, remove_trend, remove_dc_offset, remove_high_frequency_noise, remove_low_frequency_noise
