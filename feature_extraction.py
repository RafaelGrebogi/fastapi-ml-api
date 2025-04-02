import numpy as np
import pandas as pd
import os
import datetime
import glob
from scipy.fftpack import fft
from pathlib import Path
from stat_features import compute_frequency_features, compute_time_features

def extract_features_from_firebase_batch(batch_data: dict, output_dir="data/features"):
    """
    Extract statistical features using a sliding window and save one CSV per session.

    Parameters:
        batch_data (dict): Firebase batch structured as {msg_id: {...}}
        output_dir (str): Directory to save CSV files

    Returns:
        (bool, str): Success flag and output file path
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    WINDOW_SIZE = 128
    STEP_SIZE = 64

    all_features = []

    for msg_id, content in batch_data.items():
        samples = content.get("samples", [])
        label = samples[0].get("target", "Unknown") if samples else "Unknown"

        if not samples:
            print(f"⚠️ Not enough samples in batch: {msg_id}")
            continue

        accel_map = {
            'x': 'accel_x',
            'y': 'accel_y',
            'z': 'accel_z'
        }

        gyro_map = {
            'x': 'gyro_x',
            'y': 'gyro_y',
            'z': 'gyro_z'
        }
        nSamples = len(samples)

        if nSamples < WINDOW_SIZE:
            # Case 1: Not enough samples, process full batch once
            df = pd.DataFrame(samples)
            features = {}

            for axis, col in accel_map.items():
                if col in df.columns:
                    signal = df[col].values
                    # Time domain
                    features.update(compute_time_features(signal, f'acc_{axis}'))

                    # Frequency domain
                    fft_vals = np.abs(fft(signal))[:len(signal) // 2]
                    features.update(compute_frequency_features(fft_vals, f'acc_{axis}'))

            for axis, col in gyro_map.items():
                if col in df.columns:
                    signal = df[col].values
                    # Time domain
                    features.update(compute_time_features(signal, f'gyro_{axis}'))

                    # Frequency domain
                    fft_vals = np.abs(fft(signal))[:len(signal) // 2]
                    features.update(compute_frequency_features(fft_vals, f'gyro_{axis}'))

            features["label"] = label
            features["msg_id"] = msg_id
            all_features.append(features)

        else:
            # Case 2: Enough samples, use sliding window
            for start in range(0, nSamples - WINDOW_SIZE + 1, STEP_SIZE):
                window = samples[start:start + WINDOW_SIZE]
                df = pd.DataFrame(window)
                features = {}

                for axis, col in accel_map.items():
                    if col in df.columns:
                        signal = df[col].values
                        # Time domain
                        features.update(compute_time_features(signal, f'acc_{axis}'))

                        # Frequency domain
                        fft_vals = np.abs(fft(signal))[:len(signal) // 2]
                        features.update(compute_frequency_features(fft_vals, f'acc_{axis}'))

                for axis, col in gyro_map.items():
                    if col in df.columns:
                        signal = df[col].values
                        # Time domain
                        features.update(compute_time_features(signal, f'gyro_{axis}'))

                        # Frequency domain
                        fft_vals = np.abs(fft(signal))[:len(signal) // 2]
                        features.update(compute_frequency_features(fft_vals, f'gyro_{axis}'))

                features["label"] = label
                features["msg_id"] = f"{msg_id}_w{start}"

                all_features.append(features)

    # Save final combined session file
    if not all_features:
        print("⚠️ No features extracted.")
        return False, None

    df_combined = pd.DataFrame(all_features)
    col_order = ['label'] + [col for col in df_combined.columns if col not in ['label', 'msg_id']] + ['msg_id']

    # Extract device ID from first message
    session_id = list(batch_data.keys())[0].split('_')[0]

    # Get current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine next index (based on existing files)
    existing_files = glob.glob(os.path.join(output_dir, f"{session_id}_session*.csv"))
    next_index = len(existing_files) + 1

    # Build file name with index and timestamp
    csv_filename = f"{session_id}_session{next_index}_{timestamp}.csv"
    csv_path = os.path.join(output_dir, csv_filename)

    # Save the file
    df_combined[col_order].to_csv(csv_path, index=False)
    print(f"✅ Session CSV saved: {csv_path}")

    return True, csv_path
