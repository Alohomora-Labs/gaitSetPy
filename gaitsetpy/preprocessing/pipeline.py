'''
This file contains functions for preprocessing the data.

Clipping, Noise Removal etc.

We'll have the following functions:
clip_sliding_windows, remove_noise, remove_outliers, remove_baseline, remove_drift, remove_artifacts, remove_trend, remove_dc_offset, remove_high_frequency_noise, remove_low_frequency_noise
'''

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

def clip_sliding_windows(data, min_val=-1, max_val=1):
    """
    Clip values in the sliding windows to be within a specified range.
    """
    return np.clip(data, min_val, max_val)

def remove_noise(data, window_size=5):
    """
    Apply a moving average filter to reduce noise.
    """
    return data.rolling(window=window_size, center=True).mean().fillna(method="bfill").fillna(method="ffill")

def remove_outliers(data, threshold=3):
    """
    Remove outliers beyond a given threshold using the Z-score method.
    """
    mean, std = data.mean(), data.std()
    return data[(data - mean).abs() <= threshold * std]

def remove_baseline(data):
    """
    Remove baseline by subtracting the mean.
    """
    return data - data.mean()

def remove_drift(data, cutoff=0.01, fs=100):
    """
    Remove low-frequency drift using a high-pass filter.
    """
    b, a = butter(1, cutoff / (fs / 2), btype='highpass')
    return filtfilt(b, a, data)

def remove_artifacts(data, method="interpolate"):
    """
    Remove artifacts by interpolating missing values.
    """
    return data.interpolate(method="linear").fillna(method="bfill").fillna(method="ffill")

def remove_trend(data, order=2):
    """
    Remove trends using polynomial fitting.
    """
    x = np.arange(len(data))
    poly_coeffs = np.polyfit(x, data, order)
    trend = np.polyval(poly_coeffs, x)
    return data - trend

def remove_dc_offset(data):
    """
    Remove DC offset by subtracting the mean.
    """
    return data - data.mean()

def remove_high_frequency_noise(data, cutoff=10, fs=100):
    """
    Apply a low-pass filter to remove high-frequency noise.
    """
    b, a = butter(1, cutoff / (fs / 2), btype='lowpass')
    return filtfilt(b, a, data)

def remove_low_frequency_noise(data, cutoff=0.5, fs=100):
    """
    Apply a high-pass filter to remove low-frequency noise.
    """
    b, a = butter(1, cutoff / (fs / 2), btype='highpass')
    return filtfilt(b, a, data)
