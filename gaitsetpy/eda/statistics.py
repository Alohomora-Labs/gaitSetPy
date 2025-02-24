'''
This module contains functions for statistical analysis of sensor data.

Maintainer: @aharshit123456
'''

import matplotlib.pyplot as plt
import numpy as np

def plot_sensor_with_features(sliding_windows, features, start_idx, end_idx, sensor_name="shank", num_windows=10, save=False):
    """
    @brief Plots sliding windows of a sensor's time series data with overlaid statistical features.

    This function plots the first `num_windows` sliding windows within the given `start_idx` and `end_idx`
    for a specified sensor and overlays feature values at their corresponding time indices. 
    It also displays entropy and dominant frequency in a separate plot.

    @param[in] sliding_windows List of dictionaries, where each dictionary contains:
                   - 'name': sensor name (str)
                   - 'data': List of time-series windows (each as a Pandas Series)
    @param[in] features List of dictionaries, where each dictionary contains:
                   - 'name': sensor name (str)
                   - 'features': Dictionary of extracted feature lists
    @param[in] start_idx Start index of the time window to be plotted.
    @param[in] end_idx End index of the time window to be plotted.
    @param[in] sensor_name Name of the sensor to be plotted (default: "shank").
    @param[in] num_windows Number of sliding windows to plot (default: 10).
    @param[in] save If True, saves the plot to a file instead of displaying it.

    @return None
    """

    fig, axes = plt.subplots(2, 1, figsize=(20, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Extract sensor windows
    sensor_windows = next((sw['data'] for sw in sliding_windows if sw['name'] == sensor_name), None)
    if sensor_windows is None:
        print(f"Sensor '{sensor_name}' not found in sliding_windows.")
        return

    # Extract corresponding features
    sensor_features = next((feat['features'] for feat in features if feat['name'] == sensor_name), None)
    if sensor_features is None:
        print(f"Sensor '{sensor_name}' not found in features.")
        return

    # Filter windows based on start_idx and end_idx
    filtered_windows = [series for series in sensor_windows if start_idx <= series.index[0] and series.index[-1] <= end_idx]
    
    if not filtered_windows:
        print(f"No windows found in the specified index range ({start_idx} - {end_idx}).")
        return

    # Store entropy & frequency features for separate plotting
    entropy_values = []
    dominant_frequencies = []

    # Plot first `num_windows` windows
    for i in range(min(num_windows, len(filtered_windows))):
        series = filtered_windows[i]  # Each window is a Pandas Series

        # Extract time and signal values
        time_values = series.index.to_numpy()  # Time is the index
        signal_values = series.values  # Sensor readings

        # Determine actual start and end indices for this window
        window_start, window_end = time_values[0], time_values[-1]

        # Plot time series data
        axes[0].plot(time_values, signal_values, alpha=0.6)

        # Mark start and end of each window with vertical dotted lines
        axes[0].axvline(x=window_start, color='black', linestyle='dotted', alpha=0.7)
        axes[0].axvline(x=window_end, color='black', linestyle='dotted', alpha=0.7)

        # Overlay statistical features
        for feature, marker in zip(['mean', 'rms', 'peak_height', 'mode', 'median'], ['x', 'o', 'v', '<', '^']):
            if feature in sensor_features and len(sensor_features[feature]) > i:
                feature_value = sensor_features[feature][i]
                if feature_value != 0:  # Skip zero values
                    closest_index = np.argmin(np.abs(signal_values - feature_value))
                    closest_time = time_values[closest_index]
                    axes[0].scatter(closest_time, feature_value, color='red', marker=marker, s=100)

        # Store entropy & frequency features for separate plotting
        if 'entropy' in sensor_features and len(sensor_features['entropy']) > i:
            entropy_values.append(sensor_features['entropy'][i])
        if 'dominant_frequency' in sensor_features and len(sensor_features['dominant_frequency']) > i:
            dominant_frequencies.append(sensor_features['dominant_frequency'][i])

    # Labels and title for time-series plot
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel(f'{sensor_name} Signal')
    axes[0].set_title(f'First {num_windows} windows of {sensor_name} in range {start_idx}-{end_idx} with Features')

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

    # Save or show plot
    if save:
        file_path = input("Enter the file path to save the plot (e.g., 'plot.png'): ")
        plt.savefig(file_path, dpi=300)
        print(f"Plot saved at {file_path}")
    else:
        plt.show()
