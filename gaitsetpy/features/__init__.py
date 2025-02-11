'''
This module provides functions to calculate gait features from time series data.
Maintainer: @aharshit123456
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
    calculate_auto_regression_coefficients
)

from .gait_features import (
    get_mean_for_windows,
    get_standard_deviation_for_windows,
    get_variance_for_windows,
    get_skewness_for_windows,
    get_kurtosis_for_windows,
    get_root_mean_square_for_windows,
    get_range_for_windows,
    get_median_for_windows,
    get_mode_for_windows,
    get_mean_absolute_value_for_windows,
    get_median_absolute_deviation_for_windows,
    get_peak_height_for_windows,
    get_stride_times_for_windows,
    get_step_times_for_windows,
    get_cadence_for_windows,
    get_freezing_index_for_windows,
    get_dominant_frequency_for_windows,
    get_peak_frequency_for_windows,
    get_power_spectral_entropy_for_windows,
    get_principal_harmonic_frequency_for_windows,
    get_entropy_for_windows,
    get_interquartile_range_for_windows,
    get_correlation_for_windows,
    get_auto_regression_coefficients_for_windows,
    extract_gait_features
)