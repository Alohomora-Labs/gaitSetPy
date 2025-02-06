'''
Data Preprocessing for Classification

Maintainer: @aharshit123456
'''
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


def preprocess_features(features):
    """
    Convert the features dictionary into X (feature matrix) and y (labels),
    ensuring all feature vectors have a consistent length.
    """
    X = []
    y = []
    feature_lengths = []  # Track feature lengths to standardize across sensors

    for sensor_dict in features:
        sensor_name = sensor_dict["name"]
        sensor_features = sensor_dict["features"]
        sensor_annotations = sensor_dict["annotations"]

        num_windows = len(sensor_annotations)  # Expected number of windows
        feature_arrays = []

        for key in sensor_features:
            feature_array = sensor_features[key]  # Extract the feature list
            feature_array = np.array(feature_array, dtype=object)  # Convert to NumPy object array

            # Ensure it's a list of equal-length vectors
            if isinstance(feature_array[0], (list, np.ndarray)):
                print(f"Fixing inconsistent feature '{key}' in sensor '{sensor_name}'.")

                # Find max length for this feature across all windows
                max_length = max(len(f) if isinstance(f, (list, np.ndarray)) else 1 for f in feature_array)
                feature_lengths.append(max_length)  # Store max feature length for later

                # Pad/truncate each feature to be the same length
                feature_array = np.array([
                    np.pad(np.ravel(f), (0, max_length - len(f)), 'constant', constant_values=0)
                    if isinstance(f, (list, np.ndarray)) else np.array([f] + [0] * (max_length - 1))
                    for f in feature_array
                ])

            # Ensure consistency in number of windows
            if len(feature_array) != num_windows:
                print(f"Skipping feature '{key}' due to mismatched length: {len(feature_array)} instead of {num_windows}.")
                continue

            feature_arrays.append(feature_array)

        if not feature_arrays:
            continue

        # Concatenate features per window
        try:
            feature_matrix = np.column_stack(feature_arrays)
        except ValueError:
            print(f"Error: Features in sensor '{sensor_name}' have inconsistent shapes. Skipping sensor.")
            continue

        X.append(feature_matrix)
        y.append(np.array(sensor_annotations))

    if not X or not y:
        raise ValueError("No valid features or labels found.")

    # **Fix: Standardize feature matrix sizes across sensors**
    max_feature_dim = max(map(lambda x: x.shape[1], X))  # Get the max feature size
    print(f"Standardizing all feature vectors to {max_feature_dim} dimensions.")

    # Pad/truncate all feature matrices to match max_feature_dim
    X = [np.pad(x, ((0, 0), (0, max_feature_dim - x.shape[1])), 'constant', constant_values=0) if x.shape[1] < max_feature_dim else x[:, :max_feature_dim] for x in X]

    # Stack all feature matrices
    X = np.vstack(X)  
    y = np.concatenate(y)

    return X, y