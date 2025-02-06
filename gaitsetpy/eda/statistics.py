'''
This module contains functions for statistical analysis of sensor data.

Maintainer: @aharshit123456
'''

import matplotlib.pyplot as plt
import numpy as np

def plot_sensor_with_features(sliding_windows, features, sensor_name="shank", num_windows=10):
    """
    Plots the first `num_windows` sliding windows for a given sensor and overlays feature values at correct time indices.
    """

    fig, axes = plt.subplots(2, 1, figsize=(20, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Extract sensor windows from sliding_windows
    sensor_windows = []
    for sensor_dict in sliding_windows:
        if sensor_dict['name'] == sensor_name:
            sensor_windows = sensor_dict['data']
            break

    if not sensor_windows:
        print(f"Sensor '{sensor_name}' not found in sliding_windows.")
        return

    # Find the corresponding features for the sensor
    sensor_features = None
    for feature_dict in features:
        if feature_dict['name'] == sensor_name:
            sensor_features = feature_dict['features']
            break

    if sensor_features is None:
        print(f"Sensor '{sensor_name}' not found in features.")
        return

    # Store entropy & frequency features for separate plotting
    entropy_values = []
    dominant_frequencies = []

    # Plot first `num_windows` windows
    for i in range(min(num_windows, len(sensor_windows))):
        series = sensor_windows[i]  # Each window is a Series

        # Extract time and signal values
        time_values = series.index.to_numpy()  # Time is the index
        signal_values = series.values  # Sensor readings

        # Time series plot (axes[0])
        axes[0].plot(time_values, signal_values, label=f'Window {i+1}', alpha=0.6)

        # Overlay statistical features
        for feature, marker in zip(['mean', 'rms', 'peak_height', 'mode', 'median'], ['x', 'o', 'v', '<', '^']):
            if feature in sensor_features and len(sensor_features[feature]) > i:
                feature_value = sensor_features[feature][i]
                if feature_value != 0:  # Skip zero values
                    closest_index = np.argmin(np.abs(signal_values - feature_value))
                    closest_time = time_values[closest_index]
                    axes[0].scatter(closest_time, feature_value, color='red', marker=marker, s=100, label=f'{feature} {i+1}' if i == 0 else "")

        # Standard deviation shading
        if 'std_dev' in sensor_features and len(sensor_features['std_dev']) > i:
            std_value = sensor_features['std_dev'][i]
            axes[0].fill_between(time_values, signal_values - std_value, signal_values + std_value, color='gray', alpha=0.2)

        # Store entropy & frequency features for separate plotting
        if 'entropy' in sensor_features and len(sensor_features['entropy']) > i:
            entropy_values.append(sensor_features['entropy'][i])
        if 'dominant_frequency' in sensor_features and len(sensor_features['dominant_frequency']) > i:
            dominant_frequencies.append(sensor_features['dominant_frequency'][i])

    # Labels and legends for time-series plot
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel(f'{sensor_name} Signal')
    axes[0].set_title(f'First {num_windows} windows of {sensor_name} with Features')
    axes[0].legend()

    # Frequency-domain & entropy plot (axes[1])
    if dominant_frequencies:
        window_indices = list(range(len(dominant_frequencies)))
        axes[1].plot(window_indices, dominant_frequencies, label="Dominant Frequency", marker="o", linestyle="dashed", color="blue")
    
    if entropy_values:
        axes[1].bar(window_indices, entropy_values, alpha=0.6, label="Entropy", color="green")

    axes[1].set_xlabel("Window Index")
    axes[1].set_ylabel("Feature Value")
    axes[1].set_title("Frequency & Entropy Features")
    axes[1].legend()

    plt.tight_layout()
    plt.show()