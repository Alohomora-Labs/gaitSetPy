"""
Preprocessing Pipeline for GaitSetPy

This module provides a class-based preprocessing pipeline for gait data.
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

@dataclass
class PreprocessingStep:
    """Configuration for a preprocessing step."""
    name: str
    enabled: bool = True
    params: Dict[str, Any] = None

class PreprocessingPipeline:
    """Pipeline for preprocessing gait data."""
    
    def __init__(self):
        """Initialize the preprocessing pipeline with default steps."""
        self.steps = [
            PreprocessingStep("clip", params={"min_val": -1, "max_val": 1}),
            PreprocessingStep("remove_noise"),
            PreprocessingStep("remove_outliers"),
            PreprocessingStep("remove_baseline"),
            PreprocessingStep("remove_drift"),
            PreprocessingStep("remove_artifacts"),
            PreprocessingStep("remove_trend"),
            PreprocessingStep("remove_dc_offset"),
            PreprocessingStep("remove_high_frequency_noise"),
            PreprocessingStep("remove_low_frequency_noise")
        ]
        
    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Apply all enabled preprocessing steps to the data.
        
        Args:
            data (pd.DataFrame): Input data
            **kwargs: Additional parameters for preprocessing steps
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        processed_data = data.copy()
        
        for step in self.steps:
            if step.enabled:
                processed_data = self._apply_step(processed_data, step, **kwargs)
                
        return processed_data
    
    def _apply_step(self, data: pd.DataFrame, step: PreprocessingStep, **kwargs) -> pd.DataFrame:
        """Apply a single preprocessing step."""
        method = getattr(self, f"_{step.name}", None)
        if method is None:
            raise ValueError(f"Unknown preprocessing step: {step.name}")
            
        params = {**(step.params or {}), **kwargs}
        return method(data, **params)
    
    def _clip(self, data: pd.DataFrame, min_val: float = -1, max_val: float = 1, **kwargs) -> pd.DataFrame:
        """Clip values to specified range."""
        return data.clip(min_val, max_val, axis=1)
    
    def _remove_noise(self, data: pd.DataFrame, window_size: int = 5, **kwargs) -> pd.DataFrame:
        """Remove noise using moving average."""
        return data.rolling(window=window_size, center=True).mean().fillna(method='bfill').fillna(method='ffill')
    
    def _remove_outliers(self, data: pd.DataFrame, threshold: float = 3, **kwargs) -> pd.DataFrame:
        """Remove outliers using z-score method."""
        z_scores = ((data - data.mean()) / data.std()).abs()
        return data.mask(z_scores > threshold, data.mean())
    
    def _remove_baseline(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Remove baseline by subtracting mean."""
        return data - data.mean()
    
    def _remove_drift(self, data: pd.DataFrame, poly_order: int = 2, **kwargs) -> pd.DataFrame:
        """Remove polynomial drift."""
        x = np.arange(len(data))
        for col in data.columns:
            coeffs = np.polyfit(x, data[col], poly_order)
            drift = np.polyval(coeffs, x)
            data[col] = data[col] - drift
        return data
    
    def _remove_artifacts(self, data: pd.DataFrame, threshold: float = 5, **kwargs) -> pd.DataFrame:
        """Remove sudden artifacts using median filter."""
        return data.rolling(window=3, center=True).median().fillna(method='bfill').fillna(method='ffill')
    
    def _remove_trend(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Remove linear trend."""
        return data - data.expanding().mean()
    
    def _remove_dc_offset(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Remove DC offset."""
        return data - data.mean()
    
    def _remove_high_frequency_noise(self, data: pd.DataFrame, cutoff: float = 0.1, **kwargs) -> pd.DataFrame:
        """Remove high frequency noise using butterworth filter."""
        nyquist = 0.5
        b, a = butter(4, cutoff/nyquist, btype='low')
        return pd.DataFrame(filtfilt(b, a, data), index=data.index, columns=data.columns)
    
    def _remove_low_frequency_noise(self, data: pd.DataFrame, cutoff: float = 0.01, **kwargs) -> pd.DataFrame:
        """Remove low frequency noise using butterworth filter."""
        nyquist = 0.5
        b, a = butter(4, cutoff/nyquist, btype='high')
        return pd.DataFrame(filtfilt(b, a, data), index=data.index, columns=data.columns)
    
    def enable_step(self, step_name: str) -> None:
        """Enable a preprocessing step."""
        for step in self.steps:
            if step.name == step_name:
                step.enabled = True
                return
        raise ValueError(f"Unknown preprocessing step: {step_name}")
    
    def disable_step(self, step_name: str) -> None:
        """Disable a preprocessing step."""
        for step in self.steps:
            if step.name == step_name:
                step.enabled = False
                return
        raise ValueError(f"Unknown preprocessing step: {step_name}")
    
    def configure_step(self, step_name: str, params: Dict[str, Any]) -> None:
        """Configure parameters for a preprocessing step."""
        for step in self.steps:
            if step.name == step_name:
                step.params = {**(step.params or {}), **params}
                return
        raise ValueError(f"Unknown preprocessing step: {step_name}")
