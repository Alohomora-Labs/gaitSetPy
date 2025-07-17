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
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import sys
import os
import numpy as np
# sys.path.append(r"/Users/harshit/Desktop/coding/alohomora_labs/gaitSetPy/")
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

    # Helper to ensure scalar values
    def to_scalar(x):
        if isinstance(x, (list, np.ndarray)):
            if len(x) == 0:
                return 0.0
            return float(np.mean(x))
        return float(x)

    # Transform features_data into the required structure for model training
    feature_dicts = []
    for rec in features_data:
        name = rec["name"]
        features_list = rec["features"]
        if not features_list:
            continue
        # Get all feature keys except 'sensor' and 'label'
        feature_keys = [k for k in features_list[0].keys() if k not in ("sensor", "label")]
        features = {k: [to_scalar(f.get(k, 0)) for f in features_list] for k in feature_keys}
        annotations = [f["label"] for f in features_list if "label" in f]
        feature_dicts.append({
            "name": name,
            "features": features,
            "annotations": annotations
        })

    # Impute NaNs and ensure all features are numeric arrays
    for fd in feature_dicts:
        for k, arr in fd['features'].items():
            arr_np = np.array(arr, dtype=np.float32)
            if np.any(np.isnan(arr_np)):
                arr_np = np.nan_to_num(arr_np, nan=0.0)
            fd['features'][k] = arr_np.tolist()

    # For legacy code and reporting
    X = []
    y = []
    for feature_set in feature_dicts:
        features = feature_set.get('features', {})
        annotations = feature_set.get('annotations', [])
        # Transpose features to get per-window vectors
        if features:
            feature_keys = list(features.keys())
            for i in range(len(annotations)):
                feature_vector = [to_scalar(features[k][i]) for k in feature_keys]
                X.append(feature_vector)
                y.append(annotations[i])
    # Pad for reporting
    if X:
        try:
            X = np.array(X, dtype=np.float32)
            y = np.array(y)
            max_length = max(len(x) for x in X)
            X_padded = [x.tolist() + [0]*(max_length - len(x)) for x in X]
            X = np.array(X_padded, dtype=np.float32)
            print(f"\nPrepared {X.shape[0]} feature vectors with {X.shape[1]} features each")
            print(f"Activity distribution: {np.unique(y, return_counts=True)}")
        except Exception as e:
            print(f"Reporting skipped due to error: {e}")
            X = None
            y = None
    else:
        X = None
        y = None

    # --- TEST ALL MODELS ---
    from gaitsetpy.classification.models import get_classification_model
    model_names = ['random_forest', 'mlp', 'lstm', 'bilstm', 'cnn', 'gnn']
    print("\nTesting all classification models:\n")
    for model_name in model_names:
        print(f"\n--- {model_name.upper()} ---")
        # Model-specific kwargs
        kwargs = {}
        if X is not None and hasattr(X, 'shape') and y is not None:
            if model_name == 'cnn':
                kwargs['input_channels'] = X.shape[1]
            if model_name in ['lstm', 'bilstm']:
                kwargs['input_size'] = X.shape[1]
            if model_name in ['cnn', 'lstm', 'bilstm']:
                kwargs['num_classes'] = len(np.unique(y))
            if model_name == 'gnn':
                kwargs['input_dim'] = X.shape[1]
                kwargs['output_dim'] = len(np.unique(y))
        try:
            model = get_classification_model(model_name, **{k: v for k, v in kwargs.items() if v is not None})
        except Exception as e:
            print(f"Could not instantiate {model_name}: {e}")
            continue
        # For GNN, need adjacency matrix matching total number of windows
        train_kwargs = {}
        eval_kwargs = {}
        if model_name == 'gnn':
            total_windows = sum(len(fd['annotations']) for fd in feature_dicts)
            adj = np.eye(total_windows, dtype=np.float32)
            train_kwargs['adjacency_matrix'] = adj
            eval_kwargs['adjacency_matrix'] = adj
        # Train
        try:
            model.train(feature_dicts, **train_kwargs)
        except Exception as e:
            print(f"Training failed for {model_name}: {e}")
            continue
        # Evaluate
        try:
            metrics = model.evaluate(feature_dicts, detailed_report=True, **eval_kwargs)
            print("Classification Report:")
            if 'classification_report' in metrics:
                import pprint
                pprint.pprint(metrics['classification_report'])
            else:
                print("No detailed report available.")
            print("Confusion Matrix:")
            print(np.array(metrics['confusion_matrix']))
        except Exception as e:
            print(f"Evaluation failed for {model_name}: {e}")
            continue
    print("\nAll models tested.")
    
    # Plot sample data
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()