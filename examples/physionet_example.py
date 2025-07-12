#!/usr/bin/env python3
"""
PhysioNet VGRF Dataset Example
==============================

This example demonstrates how to use the PhysioNet VGRF dataset loader
and feature extractors for gait analysis in Parkinson's disease.

The PhysioNet dataset contains vertical ground reaction force (VGRF) data
from subjects with Parkinson's disease and healthy controls.

Maintainer: @aharshit123456
"""

import os
import numpy as np
import pandas as pd
from gaitsetpy.dataset import PhysioNetLoader
from gaitsetpy.features import PhysioNetFeatureExtractor, LBPFeatureExtractor, FourierSeriesFeatureExtractor
from gaitsetpy.classification.models import RandomForestModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def load_physionet_dataset(data_dir: str = "data/physionet"):
    """
    Load the PhysioNet dataset using the PhysioNetLoader.
    
    Args:
        data_dir: Directory to store the dataset
        
    Returns:
        Tuple of (data, names, labels)
    """
    print("🔄 Loading PhysioNet dataset...")
    
    # Create dataset loader
    loader = PhysioNetLoader()
    
    # Load the dataset
    data, names = loader.load_data(data_dir)
    
    # Get labels
    labels = loader.get_labels()
    
    print(f"✅ Loaded {len(data)} files")
    print(f"📊 Dataset summary:")
    print(f"   - Control subjects: {labels.count('Co')}")
    print(f"   - Parkinson's patients: {labels.count('Pt')}")
    
    return data, names, labels, loader


def create_sliding_windows(data, names, loader, window_size=600, step_size=100):
    """
    Create sliding windows from the loaded data.
    
    Args:
        data: List of DataFrames
        names: List of names
        loader: PhysioNetLoader instance
        window_size: Size of sliding window
        step_size: Step size for sliding window
        
    Returns:
        List of sliding window dictionaries
    """
    print("\n🔄 Creating sliding windows...")
    
    windows = loader.create_sliding_windows(data, names, window_size, step_size)
    
    total_windows = sum(w['metadata']['num_windows'] for w in windows if 'metadata' in w)
    print(f"✅ Created {total_windows} total windows from {len(windows)} files")
    
    return windows


def extract_physionet_features(windows, fs=100):
    """
    Extract PhysioNet-specific features including LBP and Fourier series.
    
    Args:
        windows: List of sliding window dictionaries
        fs: Sampling frequency
        
    Returns:
        Extracted features
    """
    print("\n🔄 Extracting PhysioNet features...")
    
    # Create feature extractor
    extractor = PhysioNetFeatureExtractor(verbose=True)
    
    # Extract features for each file
    all_features = []
    
    for window_dict in windows:
        if 'windows' in window_dict:
            # Extract features for this file
            features = extractor.extract_features(window_dict['windows'], fs)
            all_features.append({
                'name': window_dict['name'],
                'features': features,
                'metadata': window_dict.get('metadata', {})
            })
    
    print(f"✅ Extracted features from {len(all_features)} files")
    
    return all_features


def prepare_classification_data(all_features):
    """
    Prepare data for classification by flattening features.
    
    Args:
        all_features: List of feature dictionaries
        
    Returns:
        Tuple of (X, y) for classification
    """
    print("\n🔄 Preparing classification data...")
    
    X = []
    y = []
    
    for file_features in all_features:
        file_name = file_features['name']
        features = file_features['features']
        metadata = file_features.get('metadata', {})
        
        # Get label from metadata or filename
        if 'label' in metadata:
            label = metadata['label']
        else:
            # Extract from filename
            label = 'Co' if 'Co' in file_name else 'Pt'
        
        # Flatten features for each sensor
        for sensor_features in features:
            sensor_name = sensor_features['name']
            sensor_data = sensor_features['features']
            
            # Create feature vector by concatenating all features
            feature_vector = []
            
            for feature_name, feature_values in sensor_data.items():
                if isinstance(feature_values, list):
                    # For list features, use mean across windows
                    if len(feature_values) > 0:
                        if isinstance(feature_values[0], (list, np.ndarray)):
                            # For nested lists (like histograms), flatten and take mean
                            flat_values = []
                            for val in feature_values:
                                if isinstance(val, (list, np.ndarray)):
                                    flat_values.extend(val)
                                else:
                                    flat_values.append(val)
                            feature_vector.append(np.mean(flat_values))
                        else:
                            feature_vector.append(np.mean(feature_values))
                    else:
                        feature_vector.append(0)
                else:
                    feature_vector.append(feature_values)
            
            if len(feature_vector) > 0:
                X.append(feature_vector)
                y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"✅ Prepared {len(X)} samples with {X.shape[1]} features each")
    print(f"📊 Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    return X, y


def train_classifier(X, y):
    """
    Train a random forest classifier.
    
    Args:
        X: Feature matrix
        y: Labels
        
    Returns:
        Trained classifier and test results
    """
    print("\n🔄 Training classifier...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    rf = RandomForestModel(verbose=True)
    rf.train(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = rf.predict(X_test_scaled)
    metrics = rf.evaluate(X_test_scaled, y_test)
    
    print(f"✅ Training completed!")
    print(f"📊 Test Accuracy: {metrics['accuracy']:.3f}")
    print(f"📊 Test Precision: {metrics['precision']:.3f}")
    print(f"📊 Test Recall: {metrics['recall']:.3f}")
    print(f"📊 Test F1-Score: {metrics['f1']:.3f}")
    
    return rf, scaler, (X_test_scaled, y_test, y_pred)


def visualize_results(y_test, y_pred):
    """
    Visualize classification results.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
    """
    print("\n🔄 Creating visualizations...")
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Control', 'Parkinson\'s'], 
                yticklabels=['Control', 'Parkinson\'s'])
    plt.title('Confusion Matrix - PhysioNet VGRF Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    
    # Classification Report
    print("\n📊 Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Control', 'Parkinson\'s']))


def main():
    """
    Main function to run the complete PhysioNet analysis pipeline.
    """
    print("🚀 PhysioNet VGRF Dataset Analysis Pipeline")
    print("="*60)
    
    # Step 1: Load dataset
    data, names, labels, loader = load_physionet_dataset()
    
    # Step 2: Create sliding windows
    windows = create_sliding_windows(data, names, loader)
    
    # Step 3: Extract features
    all_features = extract_physionet_features(windows)
    
    # Step 4: Prepare classification data
    X, y = prepare_classification_data(all_features)
    
    # Step 5: Train classifier
    rf, scaler, (X_test, y_test, y_pred) = train_classifier(X, y)
    
    # Step 6: Visualize results
    visualize_results(y_test, y_pred)
    
    print("\n🎉 Analysis completed successfully!")
    print("✅ PhysioNet VGRF dataset analysis pipeline finished.")


if __name__ == "__main__":
    main() 