import numpy as np
import pandas as pd
import os
from scipy.fftpack import fft
from scipy.stats import entropy
from pathlib import Path

def extract_features_from_firebase_batch(batch_data: dict, output_dir="data/features"):
    """
    Extract FFT-based energy and entropy from accelerometer data in a Firebase batch.
    
    Parameters:
        batch_data (dict): Firebase batch structured as {msg_id: {...}}
        output_dir (str): Directory to save CSV files

    Returns:
        (bool, str): Success flag and output directory path
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for msg_id, content in batch_data.items():
        acc = content.get("accelerometer", [])
        label = content.get("label", "Unknown")

        if not acc:
            print(f"⚠️ No accelerometer data in batch: {msg_id}")
            continue

        # Convert to DataFrame
        df = pd.DataFrame(acc)
        features = {}

        for axis in ['x', 'y', 'z']:
            if axis not in df.columns:
                continue

            signal = df[axis].values
            N = len(signal)

            # Apply FFT
            fft_vals = np.abs(fft(signal))
            fft_vals = fft_vals[:N // 2]  # Keep positive frequencies only

            # Energy
            energy = np.sum(fft_vals ** 2)

            # Spectral Entropy
            psd = fft_vals ** 2
            psd_norm = psd / np.sum(psd) if np.sum(psd) != 0 else np.ones_like(psd) / len(psd)
            spec_entropy = entropy(psd_norm, base=2)

            features[f'acc_{axis}_energy'] = energy
            features[f'acc_{axis}_entropy'] = spec_entropy

        features["label"] = label
        features["msg_id"] = msg_id

        # Save as single-row CSV
        df_out = pd.DataFrame([features])
        csv_path = os.path.join(output_dir, f"{msg_id}.csv")
        df_out.to_csv(csv_path, index=False)
        print(f"✅ Features saved: {csv_path}")

    return True, output_dir
