# test_ml_api.py

import pytest
import json
import os
import pandas as pd
from feature_extraction import extract_features_from_firebase_batch

@pytest.fixture
def single_message_data():
    return {
        "C85D60BD9E7C_20250425154651_1": {
        "calibration": {
          "cos_tilt_x": 1,
          "cos_tilt_y": 0.034904,
          "gyroBias_x": 0.192195,
          "gyroBias_y": 0.022298,
          "gyroBias_z": 0.009426,
          "sin_tilt_x": 0.000929,
          "sin_tilt_y": -0.999391
        },
        "device_id": "C85D60BD9E7C",
        "message_id": "C85D60BD9E7C_20250425154651_1",
        "samples": [
          {
            "accel_x": 0.49729,
            "accel_y": -9.966604,
            "accel_z": -0.019764,
            "gyro_x": -0.01553,
            "gyro_y": 0.007279,
            "gyro_z": 0.019885,
            "target": "Normal Walk",
            "time": "2025-04-25T15:46:51Z"
          },
          {
            "accel_x": 0.520985,
            "accel_y": -9.985736,
            "accel_z": -0.027778,
            "gyro_x": -0.016863,
            "gyro_y": 0.004348,
            "gyro_z": 0.020684,
            "target": "Normal Walk",
            "time": "2025-04-25T15:46:51Z"
          },
          {
            "accel_x": 0.541875,
            "accel_y": -10.01205,
            "accel_z": -0.047673,
            "gyro_x": -0.01633,
            "gyro_y": 0.00275,
            "gyro_z": 0.020418,
            "target": "Normal Walk",
            "time": "2025-04-25T15:46:51Z"
          },
          {
            "accel_x": 0.57343,
            "accel_y": -10.07906,
            "accel_z": -0.106271,
            "gyro_x": -0.001141,
            "gyro_y": -0.007909,
            "gyro_z": 0.010026,
            "target": "Normal Walk",
            "time": "2025-04-25T15:46:51Z"
          },
          {
            "accel_x": 0.580679,
            "accel_y": -10.06469,
            "accel_z": -0.104128,
            "gyro_x": 0.016445,
            "gyro_y": -0.008175,
            "gyro_z": 0.000167,
            "target": "Normal Walk",
            "time": "2025-04-25T15:46:51Z"
          },
          {
            "accel_x": 0.582092,
            "accel_y": -9.966524,
            "accel_z": -0.061056,
            "gyro_x": 0.015912,
            "gyro_y": -0.003379,
            "gyro_z": -0.003564,
            "target": "Normal Walk",
            "time": "2025-04-25T15:46:51Z"
          },
          {
            "accel_x": 0.570526,
            "accel_y": -9.944987,
            "accel_z": -0.048674,
            "gyro_x": 0.01458,
            "gyro_y": -0.001247,
            "gyro_z": -0.003564,
            "target": "Normal Walk",
            "time": "2025-04-25T15:46:51Z"
          },
          {
            "accel_x": 0.56384,
            "accel_y": -9.935417,
            "accel_z": -0.034066,
            "gyro_x": 0.014313,
            "gyro_y": -0.000714,
            "gyro_z": -0.001432,
            "target": "Normal Walk",
            "time": "2025-04-25T15:46:51Z"
          },
          {
            "accel_x": 0.561964,
            "accel_y": -9.952179,
            "accel_z": -0.019627,
            "gyro_x": 0.01378,
            "gyro_y": 0.001151,
            "gyro_z": 0.001499,
            "target": "Normal Walk",
            "time": "2025-04-25T15:46:51Z"
          },
          {
            "accel_x": 0.560962,
            "accel_y": -9.952179,
            "accel_z": -0.04834,
            "gyro_x": 0.011382,
            "gyro_y": 0.005148,
            "gyro_z": 0.006828,
            "target": "Normal Walk",
            "time": "2025-04-25T15:46:51Z"
          }
        ],
        "timestamp": "2025-04-25T15:46:51Z"
      }
    }
#         "C85D60BD9E7C_20250401103649_13": {
#             "device_id": "C85D60BD9E7C",
#             "message_id": "C85D60BD9E7C_20250401103649_13",
#             "samples": [
#                 {
#                     "accel_x": 0.684684,
#                     "accel_y": -0.095362,
#                     "accel_z": 9.684546,
#                     "gyro_x": 0.000184,
#                     "gyro_y": -0.000365,
#                     "gyro_z": -0.000443,
#                     "target": "Normal Walk",
#                     "time": "2025-04-01T10:36:49Z"
#                 },
#                 {
#                     "accel_x": 0.689472,
#                     "accel_y": -0.102545,
#                     "accel_z": 9.679758,
#                     "gyro_x": -0.000083,
#                     "gyro_y": -0.000099,
#                     "gyro_z": -0.000176,
#                     "target": "Normal Walk",
#                     "time": "2025-04-01T10:36:49Z"
#                 },
#                 {
#                     "accel_x": 0.699049,
#                     "accel_y": -0.104939,
#                     "accel_z": 9.679758,
#                     "gyro_x": -0.000083,
#                     "gyro_y": -0.000365,
#                     "gyro_z": 0.00009,
#                     "target": "Normal Walk",
#                     "time": "2025-04-01T10:36:49Z"
#                 },
#                 {
#                     "accel_x": 0.703837,
#                     "accel_y": -0.102545,
#                     "accel_z": 9.684546,
#                     "gyro_x": -0.000083,
#                     "gyro_y": -0.000099,
#                     "gyro_z": 0.00009,
#                     "target": "Normal Walk",
#                     "time": "2025-04-01T10:36:49Z"
#                 },
#                 {
#                     "accel_x": 0.69426,
#                     "accel_y": -0.104939,
#                     "accel_z": 9.686941,
#                     "gyro_x": -0.000083,
#                     "gyro_y": -0.000099,
#                     "gyro_z": 0.00009,
#                     "target": "Normal Walk",
#                     "time": "2025-04-01T10:36:49Z"
#                 },
#                 {
#                     "accel_x": 0.689472,
#                     "accel_y": -0.104939,
#                     "accel_z": 9.682153,
#                     "gyro_x": -0.000083,
#                     "gyro_y": -0.000099,
#                     "gyro_z": 0.00009,
#                     "target": "Normal Walk",
#                     "time": "2025-04-01T10:36:49Z"
#                 },
#                 {
#                     "accel_x": 0.689472,
#                     "accel_y": -0.104939,
#                     "accel_z": 9.684546,
#                     "gyro_x": -0.000083,
#                     "gyro_y": -0.000365,
#                     "gyro_z": 0.000357,
#                     "target": "Normal Walk",
#                     "time": "2025-04-01T10:36:49Z"
#                 },
#                 {
#                     "accel_x": 0.689472,
#                     "accel_y": -0.107333,
#                     "accel_z": 9.694123,
#                     "gyro_x": -0.000083,
#                     "gyro_y": -0.000099,
#                     "gyro_z": 0.00009,
#                     "target": "Normal Walk",
#                     "time": "2025-04-01T10:36:49Z"
#                 },
#                 {
#                     "accel_x": 0.682289,
#                     "accel_y": -0.104938,
#                     "accel_z": 9.696518,
#                     "gyro_x": 0.000184,
#                     "gyro_y": 0.000168,
#                     "gyro_z": -0.000176,
#                     "target": "Normal Walk",
#                     "time": "2025-04-01T10:36:49Z"
#                 },
#                 {
#                     "accel_x": 0.687078,
#                     "accel_y": -0.097756,
#                     "accel_z": 9.689334,
#                     "gyro_x": -0.000083,
#                     "gyro_y": 0.000168,
#                     "gyro_z": -0.000176,
#                     "target": "Normal Walk",
#                     "time": "2025-04-01T10:36:49Z"
#                 }
#             ],
#             "timestamp": "2025-04-01T10:36:49Z"
#         }
    # }
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------
def test_extract_features_single_message(single_message_data):
    success, output_dir = extract_features_from_firebase_batch(single_message_data)
    assert success, "Feature extraction failed."
    assert output_dir is not None and isinstance(output_dir, str)

#------------------------------------------------------------------------------------
def test_multi_message_extraction_from_file():
    with open("/mnt/c/Users/rafag/ml-api/data/esp32_rtdb_filtered.json", "r") as f:
        test_data = json.load(f)

    success, output_path = extract_features_from_firebase_batch(test_data, USE_MULTI_MESSAGE_WINDOW=True)

    assert success, "Feature extraction failed"
    assert output_path is not None, "Output path is None"
    assert os.path.isfile(output_path), "CSV file was not created"

    df = pd.read_csv(output_path)
    assert not df.empty, "CSV file is empty"
    assert "label" in df.columns, "CSV file does not contain 'label' column"

#------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------




#------------------------------------------------------------------------------------