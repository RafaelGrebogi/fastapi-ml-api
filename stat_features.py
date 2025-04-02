import numpy as np
from scipy.stats import iqr, skew, kurtosis, entropy

def compute_frequency_features(fft_vals, prefix):
    """
    Compute statistical features in the frequency domain.
    
    Parameters:
        fft_vals (np.ndarray): Magnitude of FFT (real-valued)
        prefix (str): Feature prefix, e.g. 'acc_x'

    Returns:
        dict: Feature names and their values
    """
    features = {}

    if len(fft_vals) == 0:
        return features

    power_spectrum = fft_vals ** 2
    power_sum = np.sum(power_spectrum)
    psd_norm = power_spectrum / power_sum if power_sum != 0 else np.ones_like(power_spectrum) / len(power_spectrum)

    features[f'{prefix}_mean'] = np.mean(fft_vals)
    features[f'{prefix}_median'] = np.median(fft_vals)
    features[f'{prefix}_std'] = np.std(fft_vals)
    features[f'{prefix}_min'] = np.min(fft_vals)
    features[f'{prefix}_max'] = np.max(fft_vals)
    features[f'{prefix}_iqr'] = iqr(fft_vals)
    features[f'{prefix}_skew'] = skew(fft_vals)
    features[f'{prefix}_kurtosis'] = kurtosis(fft_vals)
    features[f'{prefix}_energy'] = power_sum
    features[f'{prefix}_entropy'] = entropy(psd_norm, base=2)

    return features

def compute_time_features(signal, prefix):
    """
    Compute statistical features in the time domain.

    Parameters:
        signal (np.ndarray): Raw signal (time domain)
        prefix (str): Feature prefix, e.g. 'acc_x'

    Returns:
        dict: Feature names and their values
    """
    features = {}

    if len(signal) == 0:
        return features

    features[f'{prefix}_mean_t'] = np.mean(signal)
    features[f'{prefix}_median_t'] = np.median(signal)
    features[f'{prefix}_std_t'] = np.std(signal)
    features[f'{prefix}_min_t'] = np.min(signal)
    features[f'{prefix}_max_t'] = np.max(signal)
    features[f'{prefix}_iqr_t'] = iqr(signal)
    features[f'{prefix}_skew_t'] = skew(signal)
    features[f'{prefix}_kurtosis_t'] = kurtosis(signal)

    return features
