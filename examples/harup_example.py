"""
HAR-UP Dataset Example

This example demonstrates how to use the HAR-UP dataset with gaitsetpy.
It shows how to:
1. Load the HAR-UP dataset (with automatic download option)
2. Create sliding windows
3. Extract features using the dedicated HAR-UP feature extractor
4. Visualize the data
5. Train a simple classifier for activity recognition

The HAR-UP dataset contains data from 17 subjects performing 11 different activities,
including normal activities (walking, sitting, etc.) and falls (forward, backward, sideward).

Note: The HAR-UP dataset is downloaded as a CSV file and automatically processed into the
required directory structure. The first time you run this example, it will download and
process the dataset, which may take some time.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import sys
import os
sys.path.append(r"/Users/harshit/Desktop/coding/alohomora_labs/gaitSetPy/")
# Import gaitsetpy modules
import gaitsetpy as gsp
from gaitsetpy.dataset import HARUPLoader, load_harup_data, create_harup_windows
from gaitsetpy.features import HARUPFeatureExtractor
from gaitsetpy.eda.visualization import plot_sensor_timeseries, plot_all_sensors, plot_activity_distribution
from gaitsetpy.eda.analyzers import harup_basic_stats, harup_missing_data_report, harup_activity_stats


def main():
    # Set data directory
    data_dir = os.path.join('data', 'harup')
    
    print("Loading HAR-UP dataset...")
    
    # Method 1: Using the class-based API
    loader = HARUPLoader()
    data, names = loader.load_data(data_dir)
    
    # Method 2: Using the legacy function API
    # data, names = load_harup_data(data_dir)
    
    if not data:
        print("No data loaded. Please make sure you've downloaded the HAR-UP dataset.")
        print("Visit https://sites.google.com/up.edu.mx/har-up/ to download the dataset.")
        return
    
    print(f"Loaded {len(data)} recordings from HAR-UP dataset")
    
    # Print information about the first recording
    print("\nFirst recording information:")
    print(f"Name: {names[0]}")
    print(f"Shape: {data[0].shape}")
    print(f"Columns: {data[0].columns.tolist()}")
    print(f"Activity: {data[0]['activity_label'].iloc[0]}")

    # --- HAR-UP ANALYSIS & VISUALIZATION ---
    print("\nBasic statistics for first subject:")
    harup_basic_stats(data[0])
    print("\nMissing data report for first subject:")
    harup_missing_data_report(data[0])
    print("\nActivity stats for first subject:")
    harup_activity_stats(data[0])
    print("\nPlotting activity distribution for first subject...")
    plot_activity_distribution(data[0])
    print("\nPlotting all sensor time series for first subject...")
    plot_all_sensors(data[0])
    print("\nPlotting a single sensor time series (BELT_ACC_X) for first subject...")
    plot_sensor_timeseries(data[0], 'BELT_ACC_X')
    
    # Create sliding windows
    print("\nCreating sliding windows...")
    window_size = 100  # 1 second at 100Hz
    step_size = 50     # 0.5 second overlap
    windows = loader.create_sliding_windows(data, names, window_size, step_size)
    
    print(f"Created {len(windows)} window sets")
    
    # Extract features
    print("\nExtracting features...")
    features_data = loader.extract_features(windows)
    
    print(f"Extracted features for {len(features_data)} recordings")
    
    # Prepare data for classification
    X = []
    y = []
    
    for feature_set in features_data:
        for feature in feature_set['features']:
            # Skip non-numeric features
            feature_vector = [v for k, v in feature.items() 
                             if k not in ['sensor', 'label'] and isinstance(v, (int, float))]
            X.append(feature_vector)
            y.append(feature['label'])
    
    # Pad sequences to have same length
    max_length = max(len(x) for x in X)
    X_padded = [x + [0]*(max_length - len(x)) for x in X]
    X = np.array(X_padded)
    y = np.array(y)
    
    print(f"\nPrepared {X.shape[0]} feature vectors with {X.shape[1]} features each")
    print(f"Activity distribution: {np.unique(y, return_counts=True)}")
    
    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train a simple classifier
    print("\nTraining a Random Forest classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate the classifier
    y_pred = clf.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('harup_confusion_matrix.png')
    print("Saved confusion matrix to 'harup_confusion_matrix.png'")
    
    # Plot sample data
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()