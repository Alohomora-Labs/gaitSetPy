"""
Visualization Module for GaitSetPy

This module provides functions for visualizing gait data and features.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

def plot_sensor_data(data: pd.DataFrame, title: Optional[str] = None) -> None:
    """
    Plot sensor data time series.
    
    Args:
        data (pd.DataFrame): Sensor data to plot
        title (Optional[str]): Plot title
    """
    plt.figure(figsize=(15, 10))
    
    # Plot each sensor
    for column in data.columns:
        if column != "annotations":
            plt.plot(data.index, data[column], label=column)
    
    plt.title(title or "Sensor Data")
    plt.xlabel("Time")
    plt.ylabel("Magnitude")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_features(features: List[Dict[str, Any]], title: Optional[str] = None) -> None:
    """
    Plot extracted features.
    
    Args:
        features (List[Dict[str, Any]]): List of feature dictionaries
        title (Optional[str]): Plot title
    """
    # Convert features to DataFrame
    feature_df = pd.DataFrame()
    for feature_dict in features:
        for feature_name, feature_values in feature_dict.items():
            if feature_name != "subject_id":
                if isinstance(feature_values, (list, np.ndarray)):
                    for i, value in enumerate(feature_values):
                        feature_df.loc[len(feature_df), f"{feature_name}_{i}"] = value
                else:
                    feature_df.loc[len(feature_df), feature_name] = feature_values
    
    # Plot correlation heatmap
    plt.figure(figsize=(15, 10))
    sns.heatmap(feature_df.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title(title or "Feature Correlations")
    plt.tight_layout()
    plt.show()
    
    # Plot feature distributions
    n_features = len(feature_df.columns)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, 5 * n_rows))
    for i, column in enumerate(feature_df.columns, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(data=feature_df[column], kde=True)
        plt.title(f"{column} Distribution")
        plt.xlabel(column)
        plt.ylabel("Count")
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[List[str]] = None) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        labels (Optional[List[str]]): Label names
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    
    if labels:
        plt.xticks(np.arange(len(labels)) + 0.5, labels, rotation=45)
        plt.yticks(np.arange(len(labels)) + 0.5, labels, rotation=0)
    
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, feature_names: List[str]) -> None:
    """
    Plot feature importance from a trained model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names (List[str]): Names of features
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model does not have feature importance scores")
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_learning_curve(train_scores: List[float], val_scores: List[float], epochs: List[int]) -> None:
    """
    Plot learning curves.
    
    Args:
        train_scores (List[float]): Training scores
        val_scores (List[float]): Validation scores
        epochs (List[int]): Epoch numbers
    """
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_scores, label='Training Score')
    plt.plot(epochs, val_scores, label='Validation Score')
    plt.title("Learning Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()