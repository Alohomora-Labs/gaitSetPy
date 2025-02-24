import numpy as np
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import welch, find_peaks
from scipy.fft import fft
from statsmodels.tsa.ar_model import AutoReg


def calculate_stride_times(signal, fs):
    """
    Calculate stride times from a signal using peak detection.
    Args:
        signal (np.array): Input signal.
        fs (int): Sampling frequency.
    Returns:
        avg_stride_time (float): Average stride time.
    """
    peaks, _ = find_peaks(signal)
    stride_times = np.diff(peaks) / fs
    avg_stride_time = np.mean(stride_times) if len(stride_times) > 0 else 0
    return avg_stride_time

def calculate_zero_crossing_rate(signal):
    """
    Calculate the zero-crossing rate of a signal.
    Args:
        signal (np.array): Input signal.
    Returns:
        zcr (float): Zero-crossing rate.
    """
    n = len(signal)
    zcr = 1 / (n - 1) * sum(0.5 * abs(np.sign(signal[i + 1]) - np.sign(signal[i])) for i in range(n - 1))
    return zcr

def calculate_power(signal, fs, band):
    """
    Calculate the power of a signal in a specific frequency band.
    Args:
        signal (np.array): Input signal.
        fs (int): Sampling frequency.
        band (tuple): Frequency band (low, high).
    Returns:
        band_power (float): Power in the specified frequency band.
    """
    # f, Pxx = welch(signal, fs=fs, nperseg=min(len(signal), 1024))
    f, Pxx = welch(signal, fs=fs, nperseg = min(len(signal), 192))  # Ensure nperseg ≤ length)
    band_power = np.trapz(Pxx[(f >= band[0]) & (f <= band[1])], f[(f >= band[0]) & (f <= band[1])])
    return band_power

def calculate_freezing_index(signal, fs):
    """
    Calculate the freezing index of a signal.
    Args:
        signal (np.array): Input signal.
        fs (int): Sampling frequency.
    Returns:
        freezing_index (float): Freezing index.
    """
    power_3_8 = calculate_power(signal, fs, (3, 8))
    power_0_5_3 = calculate_power(signal, fs, (0.5, 3))
    freezing_index = power_3_8 / power_0_5_3 if power_0_5_3 != 0 else 0
    return freezing_index

def calculate_standard_deviation(signal):
    """
    Calculate the standard deviation of a signal.
    Args:
        signal (np.array): Input signal.
    Returns:
        std_dev (float): Standard deviation.
    """
    return np.std(signal)

def calculate_entropy(signal):
    """
    Calculate the entropy of a signal.
    Args:
        signal (np.array): Input signal.
    Returns:
        entropy_value (float): Entropy.
    """
    value, counts = np.unique(signal, return_counts=True)
    probabilities = counts / len(signal)
    return entropy(probabilities, base=2)

def calculate_energy(signal):
    """
    Calculate the energy of a signal.
    Args:
        signal (np.array): Input signal.
    Returns:
        energy (float): Energy.
    """
    return np.sum(signal ** 2)

def calculate_variance(signal):
    """
    Calculate the variance of a signal.
    Args:
        signal (np.array): Input signal.
    Returns:
        variance (float): Variance.
    """
    return np.var(signal)

def calculate_kurtosis(signal):
    """
    Calculate the kurtosis of a signal.
    Args:
        signal (np.array): Input signal.
    Returns:
        kurtosis_value (float): Kurtosis.
    """
    try:
        return kurtosis(signal, fisher=False)
    except Exception as e:
        print(f"An error occurred in feature 'kurtosis': {e}")
        return 0

def calculate_step_time(signal, fs):
    """
    Calculate step times from a signal using peak detection.
    Args:
        signal (np.array): Input signal.
        fs (int): Sampling frequency.
    Returns:
        step_times (np.array): Array of step times.
    """
    peaks, _ = find_peaks(signal)
    step_times = np.diff(peaks) / fs
    return step_times

def calculate_mean(signal):
    """Calculate the mean of the signal."""
    return np.mean(signal)

def calculate_max(signal):
    """Calculate the maximum value of the signal."""
    return np.max(signal)

def calculate_min(signal):
    """Calculate the minimum value of the signal."""
    return np.min(signal)

def calculate_median(signal):
    """Calculate the median of the signal."""
    return np.median(signal)

def calculate_skewness(signal):
    """Calculate the skewness of the signal."""
    try:
        return skew(signal)
    except Exception as e:
        print(f"An error occurred in skewness: {e}")
        return 0

def calculate_root_mean_square(signal):
    """Calculate the root mean square of the signal."""
    return np.sqrt(np.mean(np.square(signal)))

def calculate_range(signal):
    """Calculate the range of the signal."""
    return np.max(signal) - np.min(signal)

def calculate_correlation(signal1, signal2):
    """Calculate the correlation between two signals."""
    return np.corrcoef(signal1, signal2)[0, 1]

def calculate_dominant_frequency(signal, fs):
    """Calculate the dominant frequency of the signal."""
    try:
        fft_values = np.abs(fft(signal))
        freqs = np.fft.fftfreq(len(signal), 1 / fs)
        dominant_freq = freqs[np.argmax(fft_values)]
        return dominant_freq
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0

def calculate_peak_height(signal):
    """Calculate the peak height of the signal."""
    peaks, _ = find_peaks(signal)
    return np.max(signal[peaks]) if len(peaks) > 0 else 0

def calculate_interquartile_range(signal):
    """Calculate the interquartile range of the signal."""
    try:
        q75, q25 = np.percentile(signal, [75, 25])
        return q75 - q25
    except Exception as e:
        print(f"An error occurred in feature 'interquartile_range': {e}")
        return 0

def calculate_mode(signal):
    """Calculate the mode of the signal."""
    values, counts = np.unique(signal, return_counts=True)
    return values[np.argmax(counts)]

def calculate_cadence(signal, fs):
    """Calculate the cadence (steps per minute) of the signal."""
    peaks, _ = find_peaks(signal)
    step_count = len(peaks)
    duration = len(signal) / fs
    return (step_count / duration) * 60

def calculate_mean_absolute_value(signal):
    """Calculate the mean absolute value of the signal."""
    return np.mean(np.abs(signal))

def calculate_median_absolute_deviation(signal):
    """Calculate the median absolute deviation of the signal."""
    return np.median(np.abs(signal - np.median(signal)))

def calculate_peak_frequency(signal, fs):
    """Calculate the peak frequency of the signal."""
    try:
        f, Pxx = welch(signal, fs=fs)
        return f[np.argmax(Pxx)]
    except Exception as e:
        print(f"An error occurred in feature 'peak_frequency': {e}")
        return 0

def calculate_peak_width(signal, fs):
    """Calculate the peak width of the signal."""
    peaks, _ = find_peaks(signal)
    if len(peaks) == 0:
        return 0
    peak_heights = signal[peaks]
    half_max = np.max(peak_heights) / 2
    widths = np.diff(np.where(signal > half_max)[0])
    return np.mean(widths) / fs if len(widths) > 0 else 0

def calculate_power_spectral_entropy(signal, fs):
    """Calculate the power spectral entropy of the signal."""
    try:
        f, Pxx = welch(signal, fs=fs)
        Pxx_norm = Pxx / np.sum(Pxx)
        return -np.sum(Pxx_norm * np.log2(Pxx_norm + np.finfo(float).eps))
    except Exception as e:
        print(f"An error occurred in feature 'power spectral entropy': {e}")
        return 0

def calculate_principal_harmonic_frequency(signal, fs):
    """Calculate the principal harmonic frequency of the signal."""
    try:
        fft_values = np.abs(fft(signal))
        freqs = np.fft.fftfreq(len(signal), 1 / fs)
        return freqs[np.argmax(fft_values)]
    except Exception as e:
        print(f"An error occurred in feature 'principal_harmonic_frequency': {e}")
        return 0

def calculate_auto_regression_coefficients(signal, order=3):
    """Calculate the auto-regression coefficients of the signal."""
    try:
        model = AutoReg(signal, lags=order)
        results = model.fit()
        return results.params
    except Exception as e:
        print(f"An error occurred in feature 'auto_regression_coefficients': {e}")
        return 0
