"""
Core module for GaitSetPy - Contains base classes and singleton managers for modular architecture.

This module provides:
- Base classes for different components (DatasetLoader, FeatureExtractor, etc.)
- Singleton managers for plugin-based architecture
- Registry system for easy extension

Maintainer: @aharshit123456
"""

from .base_classes import (
    BaseDatasetLoader,
    BaseFeatureExtractor,
    BasePreprocessor,
    BaseEDAAnalyzer,
    BaseClassificationModel
)

from .managers import (
    DatasetManager,
    FeatureManager,
    PreprocessingManager,
    EDAManager,
    ClassificationManager
)

__all__ = [
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