'''

'''

from .utils import (
    calculate_mean,
    calculate_standard_deviation,
    calculate_variance,
    calculate_skewness,
    calculate_kurtosis,
    calculate_root_mean_square,
    calculate_range,
    calculate_median,
    calculate_mode,
    calculate_mean_absolute_value,
    calculate_median_absolute_deviation,
    calculate_peak_height,
    calculate_stride_times,
    calculate_step_time,
    calculate_cadence,
    calculate_freezing_index,
    calculate_dominant_frequency,
    calculate_peak_frequency,
    calculate_power_spectral_entropy,
    calculate_principal_harmonic_frequency,
    calculate_entropy,
    calculate_interquartile_range,
    calculate_correlation,
    calculate_auto_regression_coefficients,
    calculate_zero_crossing_rate,
    calculate_energy,
)

def get_stride_times_for_windows(windows, fs):
    """
    Calculate stride times for all windows in the input.
    Args:
        windows (list): List of windows (each window is a DataFrame or array).
        fs (int): Sampling frequency.
    Returns:
        stride_times_array (list): List of stride times for each window.
    """
    stride_times_array = []
    for window in windows:
        stride_time = calculate_stride_times(window.values, fs)
        stride_times_array.append(stride_time)
    return stride_times_array

def get_zero_crossing_rates_for_windows(windows):
    """
    Calculate zero-crossing rates for all windows in the input.
    Args:
        windows (list): List of windows (each window is a DataFrame or array).
    Returns:
        zcr_array (list): List of zero-crossing rates for each window.
    """
    zcr_array = []
    for window in windows:
        zcr = calculate_zero_crossing_rate(window.values)
        zcr_array.append(zcr)
    return zcr_array

def get_freezing_indices_for_windows(windows, fs):
    """
    Calculate freezing indices for all windows in the input.
    Args:
        windows (list): List of windows (each window is a DataFrame or array).
        fs (int): Sampling frequency.
    Returns:
        freezing_indices (list): List of freezing indices for each window.
    """
    freezing_indices = []
    for window in windows:
        freezing_index = calculate_freezing_index(window.values, fs)
        freezing_indices.append(freezing_index)
    return freezing_indices

def get_standard_deviations_for_windows(windows):
    """
    Calculate standard deviations for all windows in the input.
    Args:
        windows (list): List of windows (each window is a DataFrame or array).
    Returns:
        std_devs (list): List of standard deviations for each window.
    """
    std_devs = []
    for window in windows:
        std_dev = calculate_standard_deviation(window.values)
        std_devs.append(std_dev)
    return std_devs

def get_entropies_for_windows(windows):
    """
    Calculate entropies for all windows in the input.
    Args:
        windows (list): List of windows (each window is a DataFrame or array).
    Returns:
        entropy_array (list): List of entropies for each window.
    """
    entropy_array = []
    for window in windows:
        entropy_value = calculate_entropy(window.values)
        entropy_array.append(entropy_value)
    return entropy_array

def get_energies_for_windows(windows):
    """
    Calculate energies for all windows in the input.
    Args:
        windows (list): List of windows (each window is a DataFrame or array).
    Returns:
        energy_array (list): List of energies for each window.
    """
    energy_array = []
    for window in windows:
        energy = calculate_energy(window.values)
        energy_array.append(energy)
    return energy_array

def get_variances_for_windows(windows):
    """
    Calculate variances for all windows in the input.
    Args:
        windows (list): List of windows (each window is a DataFrame or array).
    Returns:
        variance_array (list): List of variances for each window.
    """
    variance_array = []
    for window in windows:
        variance = calculate_variance(window.values)
        variance_array.append(variance)
    return variance_array

def get_kurtosis_for_windows(windows):
    """
    Calculate kurtosis values for all windows in the input.
    Args:
        windows (list): List of windows (each window is a DataFrame or array).
    Returns:
        kurtosis_array (list): List of kurtosis values for each window.
    """
    kurtosis_array = []
    for window in windows:
        kurtosis_value = calculate_kurtosis(window.values)
        kurtosis_array.append(kurtosis_value)
    return kurtosis_array

def get_step_times_for_windows(windows, fs):
    """
    Calculate step times for all windows in the input.
    Args:
        windows (list): List of windows (each window is a DataFrame or array).
        fs (int): Sampling frequency.
    Returns:
        step_times (list): List of step times for each window.
    """
    step_times = []
    for window in windows:
        step_time = calculate_step_time(window.values, fs)
        step_times.append(step_time)
    return step_times

def get_mean_for_windows(windows):
    return [calculate_mean(window.values) for window in windows]

def get_standard_deviation_for_windows(windows):
    return [calculate_standard_deviation(window.values) for window in windows]

def get_variance_for_windows(windows):
    return [calculate_variance(window.values) for window in windows]

def get_skewness_for_windows(windows):
    return [calculate_skewness(window.values) for window in windows]

def get_kurtosis_for_windows(windows):
    return [calculate_kurtosis(window.values) for window in windows]

def get_root_mean_square_for_windows(windows):
    return [calculate_root_mean_square(window.values) for window in windows]

def get_range_for_windows(windows):
    return [calculate_range(window.values) for window in windows]

def get_median_for_windows(windows):
    return [calculate_median(window.values) for window in windows]

def get_mode_for_windows(windows):
    return [calculate_mode(window.values) for window in windows]

def get_mean_absolute_value_for_windows(windows):
    return [calculate_mean_absolute_value(window.values) for window in windows]

def get_median_absolute_deviation_for_windows(windows):
    return [calculate_median_absolute_deviation(window.values) for window in windows]

def get_peak_height_for_windows(windows):
    return [calculate_peak_height(window.values) for window in windows]

def get_stride_times_for_windows(windows, fs):
    return [calculate_stride_times(window.values, fs) for window in windows]

def get_step_times_for_windows(windows, fs):
    return [calculate_step_time(window.values, fs) for window in windows]

def get_cadence_for_windows(windows, fs):
    return [calculate_cadence(window.values, fs) for window in windows]

def get_freezing_index_for_windows(windows, fs):
    return [calculate_freezing_index(window.values, fs) for window in windows]

def get_dominant_frequency_for_windows(windows, fs):
    return [calculate_dominant_frequency(window.values, fs) for window in windows]

def get_peak_frequency_for_windows(windows, fs):
    return [calculate_peak_frequency(window.values, fs) for window in windows]

def get_power_spectral_entropy_for_windows(windows, fs):
    return [calculate_power_spectral_entropy(window.values, fs) for window in windows]

def get_principal_harmonic_frequency_for_windows(windows, fs):
    return [calculate_principal_harmonic_frequency(window.values, fs) for window in windows]

def get_entropy_for_windows(windows):
    return [calculate_entropy(window.values) for window in windows]

def get_interquartile_range_for_windows(windows):
    return [calculate_interquartile_range(window.values) for window in windows]

def get_correlation_for_windows(windows):
    return [calculate_correlation(window.values[:-1], window.values[1:]) for window in windows]

def get_auto_regression_coefficients_for_windows(windows, order=3):
    return [calculate_auto_regression_coefficients(window.values, order) for window in windows]


def extract_gait_features(daphnet_windows, fs, time_domain=True, frequency_domain=True, statistical=True):
    """
    Extract gait features for all sensor windows in the daphnet_windows array.

    Args:
        daphnet_windows (list): List of 12 dictionaries (11 sensors + 1 annotation).
                                Each dictionary contains 'name' and 'data'.
        fs (int): Sampling frequency.
        time_domain (bool): Whether to include time-domain features.
        frequency_domain (bool): Whether to include frequency-domain features.
        statistical (bool): Whether to include statistical features.

    Returns:
        features (list): List of dictionaries containing 'name', 'features', and 'annotations'.
    """
    features = []

    # Extract annotations from the last dictionary
    annotation_dict = daphnet_windows[-1]
    annotations = annotation_dict["data"]  # List of annotation DataFrames

    for i, sensor_dict in enumerate(daphnet_windows[:-1]):  # Iterate over 11 sensor dictionaries
        sensor_name = sensor_dict["name"]
        sensor_windows = sensor_dict["data"]  # List of sliding window DataFrames

        sensor_features = {
            "name": sensor_name,
            "features": [],
            "annotations": []  # Store aligned annotations
        }

        # Iterate over each sliding window
        for j, window in enumerate(sensor_windows):
            window_features = {}

            if time_domain:
                window_features.update({
                    "mean": get_mean_for_windows(window),
                    "std_dev": get_standard_deviation_for_windows(window),
                    "variance": get_variance_for_windows(window),
                    "skewness": get_skewness_for_windows(window),
                    "kurtosis": get_kurtosis_for_windows(window),
                    "rms": get_root_mean_square_for_windows(window),
                    "range": get_range_for_windows(window),
                    "median": get_median_for_windows(window),
                    "mode": get_mode_for_windows(window),
                    "mav": get_mean_absolute_value_for_windows(window),
                    "mad": get_median_absolute_deviation_for_windows(window),
                    "peak_height": get_peak_height_for_windows(window),
                    "stride_time": get_stride_times_for_windows(window, fs),
                    "step_time": get_step_times_for_windows(window, fs),
                    "cadence": get_cadence_for_windows(window, fs)
                })

            if frequency_domain:
                window_features.update({
                    "freezing_index": get_freezing_index_for_windows(window, fs),
                    "dominant_frequency": get_dominant_frequency_for_windows(window, fs),
                    "peak_frequency": get_peak_frequency_for_windows(window, fs),
                    "power_spectral_entropy": get_power_spectral_entropy_for_windows(window, fs),
                    "principal_harmonic_frequency": get_principal_harmonic_frequency_for_windows(window, fs)
                })

            if statistical:
                window_features.update({
                    "entropy": get_entropy_for_windows(window),
                    "iqr": get_interquartile_range_for_windows(window),
                    "correlation": get_correlation_for_windows(window),
                    "auto_regression_coefficients": get_auto_regression_coefficients_for_windows(window)
                })

            # Store computed features
            sensor_features["features"].append(window_features)
            # Store corresponding annotation
            sensor_features["annotations"].append(annotations[j])  # Aligning annotation with window

        features.append(sensor_features)

    return features
