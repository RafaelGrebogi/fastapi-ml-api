import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# --- CONFIGURATION ---
FILE_PATH = "data/esp32-rtdb-largeExport_v25April25.json"
DATA_PATH = "/ESP32_Develop/TrainingDataset/"
TIMEZONE_OFFSET = 0  # Adjust if needed


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def reverse_accel_calibration(ax_corr, ay_corr, az_corr, sin_tilt_x, cos_tilt_x, sin_tilt_y, cos_tilt_y):
    import numpy as np

    # Construct Rx and Ry
    Rx = np.array([
        [1,        0,         0],
        [0,  cos_tilt_y,  -sin_tilt_y],
        [0,  sin_tilt_y,   cos_tilt_y]
    ])

    Ry = np.array([
        [ cos_tilt_x, 0, sin_tilt_x],
        [          0, 1,          0],
        [-sin_tilt_x, 0, cos_tilt_x]
    ])

    # Full forward rotation matrix
    R = Ry @ Rx

    # Inverse of orthogonal matrix = transpose
    R_inv = R.T

    # Apply to corrected values to obtain raw values
    corrected = np.array([ax_corr, ay_corr, az_corr])
    raw = R_inv @ corrected

    return raw[0], raw[1], raw[2]



def reverse_gyro_calibration(gx_corr, gy_corr, gz_corr, bias_x, bias_y, bias_z):
    return gx_corr + bias_x, gy_corr + bias_y, gz_corr + bias_z


def extract_samples(data_branch):
    all_samples = []
    for msg in data_branch.values():
        calib = msg.get("calibration", {})
        for s in msg.get("samples", []):
            s["session_id"] = msg.get("message_id", "").split("_")[0]
            s.update({
                "gyroBias_x": calib.get("gyroBias_x", 0),
                "gyroBias_y": calib.get("gyroBias_y", 0),
                "gyroBias_z": calib.get("gyroBias_z", 0),
                "sin_tilt_x": calib.get("sin_tilt_x", 0),
                "cos_tilt_x": calib.get("cos_tilt_x", 1),
                "sin_tilt_y": calib.get("sin_tilt_y", 0),
                "cos_tilt_y": calib.get("cos_tilt_y", 1),
            })
            all_samples.append(s)
    return pd.DataFrame(all_samples)


def plot_sensor_data(df, session_id):
    time = df["time"]

    # Calibrated acceleration
    plt.figure(figsize=(10, 5))
    plt.plot(time, df["accel_x"], label="Calibrated X")
    plt.plot(time, df["accel_y"], label="Calibrated Y")
    plt.plot(time, df["accel_z"], label="Calibrated Z")
    plt.title(f"Calibrated Acceleration - Session {session_id}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"calibrated_accel_{session_id}.png")
    plt.close()

    # Raw acceleration
    plt.figure(figsize=(10, 5))
    plt.plot(time, df["accel_x_raw"], label="Raw X")
    plt.plot(time, df["accel_y_raw"], label="Raw Y")
    plt.plot(time, df["accel_z_raw"], label="Raw Z")
    plt.title(f"Reconstructed Raw Acceleration - Session {session_id}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"raw_accel_{session_id}.png")
    plt.close()

    # Calibrated gyroscope
    plt.figure(figsize=(10, 5))
    plt.plot(time, df["gyro_x"], label="Calibrated X")
    plt.plot(time, df["gyro_y"], label="Calibrated Y")
    plt.plot(time, df["gyro_z"], label="Calibrated Z")
    plt.title(f"Calibrated Gyroscope - Session {session_id}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"calibrated_gyro_{session_id}.png")
    plt.close()

    # Raw gyroscope
    plt.figure(figsize=(10, 5))
    plt.plot(time, df["gyro_x_raw"], label="Raw X")
    plt.plot(time, df["gyro_y_raw"], label="Raw Y")
    plt.plot(time, df["gyro_z_raw"], label="Raw Z")
    plt.title(f"Reconstructed Raw Gyroscope - Session {session_id}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"raw_gyro_{session_id}.png")
    plt.close()


def main():
    raw_data = load_json(FILE_PATH)
    data_branch = raw_data.get("ESP32_Develop", {}).get("TrainingDataset", {})
    df = extract_samples(data_branch)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")

    # Reconstruct raw values
    raw_accels = df.apply(lambda row: reverse_accel_calibration(
        row["accel_x"], row["accel_y"], row["accel_z"],
        row["sin_tilt_x"], row["cos_tilt_x"],
        row["sin_tilt_y"], row["cos_tilt_y"]
    ), axis=1)
    df["accel_x_raw"], df["accel_y_raw"], df["accel_z_raw"] = zip(*raw_accels)

    raw_gyros = df.apply(lambda row: reverse_gyro_calibration(
        row["gyro_x"], row["gyro_y"], row["gyro_z"],
        row["gyroBias_x"], row["gyroBias_y"], row["gyroBias_z"]
    ), axis=1)
    df["gyro_x_raw"], df["gyro_y_raw"], df["gyro_z_raw"] = zip(*raw_gyros)

    for session_id, group in df.groupby("session_id"):
        plot_sensor_data(group, session_id)


main()
