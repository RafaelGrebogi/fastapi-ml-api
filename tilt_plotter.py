
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# --- CONFIGURATION ---
FILE_PATH = "data/esp32-rtdb-export.json"
DATA_PATH = "/ESP32_Develop/TrainingDataset/"
ARCHIVE_PATH = "/ESP32_Develop/TrainingArchive/"
SELECTED_PATH = DATA_PATH  # ← change to ARCHIVE_PATH if needed
TIMEZONE_OFFSET = 0  # Adjust if needed

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def compute_tilt_angles(ax, ay, az):
    # Sagittal (tilt X), rotation around Y
    tilt_x = np.arctan2(ax, np.sqrt(ay**2 + az**2))
    # Frontal (tilt Y), rotation around X
    tilt_y = np.arctan2(ay, np.sqrt(ax**2 + az**2))
    # Optional: full vector tilt from projection
    tilt_z = np.arctan2(np.sqrt(ax**2 + ay**2), az)
    return tilt_x, tilt_y, tilt_z

def extract_samples(data_branch):
    all_samples = []
    for msg in data_branch.values():
        samples = msg.get("samples", [])
        for s in samples:
            s["session_id"] = msg.get("message_id", "").split("_")[0]
            all_samples.append(s)
    return pd.DataFrame(all_samples)

def main():
    raw_data = load_json(FILE_PATH)
    path_key = SELECTED_PATH.strip("/").split("/")[-1]
    data_branch = raw_data.get("ESP32_Develop", {}).get(path_key, {})

    df = extract_samples(data_branch)

    # Parse timestamps
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")

    # Compute tilt angles (in degrees)
    tilts = df.apply(lambda row: compute_tilt_angles(row["accel_x"], row["accel_y"], row["accel_z"]), axis=1)
    df["tilt_x"], df["tilt_y"], df["tilt_z"] = zip(*tilts)
    df["tilt_x"] = np.degrees(df["tilt_x"])
    df["tilt_y"] = np.degrees(df["tilt_y"])
    df["tilt_z"] = np.degrees(df["tilt_z"])

    # Plot one session at a time
    for session_id, group in df.groupby("session_id"):
        plt.figure(figsize=(10, 5))
        plt.plot(group["time"], group["tilt_x"], label="Tilt X (sagittal)")
        plt.plot(group["time"], group["tilt_y"], label="Tilt Y (frontal)")
        plt.plot(group["time"], group["tilt_z"], label="Tilt Z (xy proj)", linestyle="dotted")
        plt.xlabel("Time")
        plt.ylabel("Tilt (degrees)")
        plt.title(f"Tilt Angles - Session {session_id}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"tilt_plot_{session_id}.png")
        print(f"✅ Plot saved: tilt_plot_{session_id}.png")
        plt.close()

if __name__ == "__main__":
    main()
