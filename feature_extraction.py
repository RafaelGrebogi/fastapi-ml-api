import numpy as np
import pandas as pd
import os
import datetime
import glob
from scipy.fftpack import fft
from pathlib import Path
from stat_features import compute_frequency_features, compute_time_features



def extract_window_features(window_df, accel_map, gyro_map, label=None, msg_id=None):
    features = {}

    for axis, col in accel_map.items():
        if col in window_df.columns:
            signal = window_df[col].values
            features.update(compute_time_features(signal, f'acc_{axis}'))
            fft_vals = np.abs(fft(signal))[:len(signal) // 2]
            features.update(compute_frequency_features(fft_vals, f'acc_{axis}'))

    for axis, col in gyro_map.items():
        if col in window_df.columns:
            signal = window_df[col].values
            features.update(compute_time_features(signal, f'gyro_{axis}'))
            fft_vals = np.abs(fft(signal))[:len(signal) // 2]
            features.update(compute_frequency_features(fft_vals, f'gyro_{axis}'))

    # Add label and message ID if provided
    if label is not None:
        features["label"] = label
    if msg_id is not None:
        features["msg_id"] = msg_id

    return features




def extract_features_from_firebase_batch(batch_data: dict, output_dir="data/features", USE_MULTI_MESSAGE_WINDOW=False, csv_path: str = "") -> str:

    """
    Extract statistical features using a sliding window and save one CSV per session.

    Parameters:
        batch_data (dict): Firebase batch structured as {msg_id: {...}}
        output_dir (str): Directory to save CSV files

    Returns:
        (bool, str): Success flag and output file path
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    
    # Configuration
    # USE_MULTI_MESSAGE_WINDOW = False  # ← toggle this flag

    WINDOW_SIZE = 64
    # STEP_SIZE = 64

    accel_map = {'x': 'accel_x', 'y': 'accel_y', 'z': 'accel_z'}
    gyro_map = {'x': 'gyro_x', 'y': 'gyro_y', 'z': 'gyro_z'}

    all_features = []
    if USE_MULTI_MESSAGE_WINDOW:
        full_sample_list = []

        for msg_id, content in batch_data.items():
            samples = content.get("samples", [])
            if not samples:
                print(f"⚠️ No samples found in message: {msg_id}")
                continue
            full_sample_list.extend(samples)

        if len(full_sample_list) < WINDOW_SIZE:
            print(f"⚠️ Not enough combined samples: {len(full_sample_list)}")
            return False, None

        df_all = pd.DataFrame(full_sample_list)
        current_label = df_all.iloc[0]["target"]
        window = []
        window_index = 0

        for idx, row in df_all.iterrows():
            if row["target"] == current_label:
                window.append(row)
                if len(window) == WINDOW_SIZE:
                    # Enough samples for a full window
                    window_df = pd.DataFrame(window)
                    # features = extract_window_features(window_df, accel_map, gyro_map)
                    features = extract_window_features(window_df, accel_map, gyro_map, label=current_label, msg_id=f"window_{window_index}")

                    # features["label"] = current_label
                    # features["msg_id"] = f"window_{window_index}"
                    all_features.append(features)
                    window = []  # Start new window
                    window_index += 1
            else:
                # Label changed mid-window → process what we have (even if small)
                if len(window) > 10:  # Only save windows with enough samples (optional threshold)
                    window_df = pd.DataFrame(window)
                    features = extract_window_features(window_df, accel_map, gyro_map, label=current_label, msg_id=f"window_{window_index}")

                    # features = extract_window_features(window_df, accel_map, gyro_map)
                    # features["label"] = current_label
                    # features["msg_id"] = f"window_{window_index}"
                    all_features.append(features)
                    window_index += 1

                # Start a new window with the new label
                window = [row]
                current_label = row["target"]

        # After loop ends, handle leftover samples
        if len(window) > 10:
            window_df = pd.DataFrame(window)
            features = extract_window_features(window_df, accel_map, gyro_map, label=current_label, msg_id=f"window_{window_index}")

            # features = extract_window_features(window_df, accel_map, gyro_map)
            # features["label"] = current_label
            # features["msg_id"] = f"window_{window_index}"
            all_features.append(features)


    else:
        # --- Original per-message logic ---
        for msg_id, content in batch_data.items():
            samples = content.get("samples", [])
            label = samples[0].get("target", "Unknown") if samples else "Unknown"

            if not samples:
                print(f"⚠️ Not enough samples in batch: {msg_id}")
                continue

            nSamples = len(samples)
            df = pd.DataFrame(samples)

           
            # Fixed window (no sliding)
            features = {}

             # Fixed window (no sliding)
            features = extract_window_features(df, accel_map, gyro_map, label=label, msg_id=msg_id)
            # features = extract_window_features(df, accel_map, gyro_map)
            
            # Then continue saving features as you already do
            # features["label"] = label  # (from your batch_data)
            # features["msg_id"] = msg_id
            all_features.append(features)

            

    # --- Save output CSV ---
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
    
    
    
    
    
    
    
    # WINDOW_SIZE = 128
    # STEP_SIZE = 64

    # all_features = []
    # full_sample_list = []
    # labels = []

    # accel_map = {
    #     'x': 'accel_x',
    #     'y': 'accel_y',
    #     'z': 'accel_z'
    # }

    # gyro_map = {
    #     'x': 'gyro_x',
    #     'y': 'gyro_y',
    #     'z': 'gyro_z'
    # }

    # for msg_id, content in batch_data.items():
    #     samples = content.get("samples", [])
    #     label = samples[0].get("target", "Unknown") if samples else "Unknown"

    #     if not samples:
    #         print(f"⚠️ Not enough samples in batch: {msg_id}")
    #         continue

    #     full_sample_list.extend(samples)
    #     labels.append(label)

    #     # accel_map = {
    #     #     'x': 'accel_x',
    #     #     'y': 'accel_y',
    #     #     'z': 'accel_z'
    #     # }

    #     # gyro_map = {
    #     #     'x': 'gyro_x',
    #     #     'y': 'gyro_y',
    #     #     'z': 'gyro_z'
    #     # }
    #     # nSamples = len(samples)

    #     total_samples = len(full_sample_list)

    #     if total_samples == 0:
    #         print("⚠️ No valid samples found in batch.")
    #         return False, None
        
    #     df_all = pd.DataFrame(full_sample_list)
    #     label = max(set(labels), key=labels.count)
    #     session_id = list(batch_data.keys())[0].split('_')[0]

    #     if len(batch_data) == 1 or total_samples < WINDOW_SIZE:
    #         # Case 1: Not enough samples, process full batch once
    #         print(f"ℹ️ Using FIXED extraction (samples: {total_samples})")
    #         features = {}

    #         for axis, col in accel_map.items():
    #             if col in df_all.columns:
    #                 signal = df_all[col].values
    #                 # Time domain
    #                 features.update(compute_time_features(signal, f'acc_{axis}'))

    #                 # Frequency domain
    #                 fft_vals = np.abs(fft(signal))[:len(signal) // 2]
    #                 features.update(compute_frequency_features(fft_vals, f'acc_{axis}'))

    #         for axis, col in gyro_map.items():
    #             if col in df_all.columns:
    #                 signal = df_all[col].values
    #                 # Time domain
    #                 features.update(compute_time_features(signal, f'gyro_{axis}'))

    #                 # Frequency domain
    #                 fft_vals = np.abs(fft(signal))[:len(signal) // 2]
    #                 features.update(compute_frequency_features(fft_vals, f'gyro_{axis}'))

    #         features["label"] = label
    #         features["msg_id"] = msg_id
    #         all_features.append(features)

    #     else:
    #         # Case 2: Enough samples, use sliding window
    #         print(f"ℹ️ Using SLIDING WINDOW extraction (samples: {total_samples})")
    #         for start in range(0, total_samples - WINDOW_SIZE + 1, STEP_SIZE):
    #             window = df_all.iloc[start:start + WINDOW_SIZE]
    #             df = pd.DataFrame(window)
    #             features = {}

    #             for axis, col in accel_map.items():
    #                 if col in window.columns:
    #                     signal = window[col].values
    #                     # Time domain
    #                     features.update(compute_time_features(signal, f'acc_{axis}'))

    #                     # Frequency domain
    #                     fft_vals = np.abs(fft(signal))[:len(signal) // 2]
    #                     features.update(compute_frequency_features(fft_vals, f'acc_{axis}'))

    #             for axis, col in gyro_map.items():
    #                 if col in window.columns:
    #                     signal = window[col].values
    #                     # Time domain
    #                     features.update(compute_time_features(signal, f'gyro_{axis}'))

    #                     # Frequency domain
    #                     fft_vals = np.abs(fft(signal))[:len(signal) // 2]
    #                     features.update(compute_frequency_features(fft_vals, f'gyro_{axis}'))

    #             features["label"] = label
    #             features["msg_id"] = f"{msg_id}_w{start}"

    #             all_features.append(features)

    # # Save final combined session file
    # if not all_features:
    #     print("⚠️ No features extracted.")
    #     return False, None

    # df_combined = pd.DataFrame(all_features)
    # col_order = ['label'] + [col for col in df_combined.columns if col not in ['label', 'msg_id']] + ['msg_id']

    # # Extract device ID from first message
    # session_id = list(batch_data.keys())[0].split('_')[0]

    # # Get current timestamp
    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # # Determine next index (based on existing files)
    # existing_files = glob.glob(os.path.join(output_dir, f"{session_id}_session*.csv"))
    # next_index = len(existing_files) + 1

    # # Build file name with index and timestamp
    # csv_filename = f"{session_id}_session{next_index}_{timestamp}.csv"
    # csv_path = os.path.join(output_dir, csv_filename)

    # # Save the file
    # df_combined[col_order].to_csv(csv_path, index=False)
    # print(f"✅ Session CSV saved: {csv_path}")

    # return True, csv_path