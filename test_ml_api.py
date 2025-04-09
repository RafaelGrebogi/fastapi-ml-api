# test_ml_api.py

import pytest
import json
import os
import pandas as pd
from feature_extraction import extract_features_from_firebase_batch

@pytest.fixture
def single_message_data():
    return {
        "C85D60BD9E7C_20250401103649_13": {
            "device_id": "C85D60BD9E7C",
            "message_id": "C85D60BD9E7C_20250401103649_13",
            "samples": [
                {
                    "accel_x": 0.684684,
                    "accel_y": -0.095362,
                    "accel_z": 9.684546,
                    "gyro_x": 0.000184,
                    "gyro_y": -0.000365,
                    "gyro_z": -0.000443,
                    "target": "Normal Walk",
                    "time": "2025-04-01T10:36:49Z"
                },
                {
                    "accel_x": 0.689472,
                    "accel_y": -0.102545,
                    "accel_z": 9.679758,
                    "gyro_x": -0.000083,
                    "gyro_y": -0.000099,
                    "gyro_z": -0.000176,
                    "target": "Normal Walk",
                    "time": "2025-04-01T10:36:49Z"
                },
                {
                    "accel_x": 0.699049,
                    "accel_y": -0.104939,
                    "accel_z": 9.679758,
                    "gyro_x": -0.000083,
                    "gyro_y": -0.000365,
                    "gyro_z": 0.00009,
                    "target": "Normal Walk",
                    "time": "2025-04-01T10:36:49Z"
                },
                {
                    "accel_x": 0.703837,
                    "accel_y": -0.102545,
                    "accel_z": 9.684546,
                    "gyro_x": -0.000083,
                    "gyro_y": -0.000099,
                    "gyro_z": 0.00009,
                    "target": "Normal Walk",
                    "time": "2025-04-01T10:36:49Z"
                },
                {
                    "accel_x": 0.69426,
                    "accel_y": -0.104939,
                    "accel_z": 9.686941,
                    "gyro_x": -0.000083,
                    "gyro_y": -0.000099,
                    "gyro_z": 0.00009,
                    "target": "Normal Walk",
                    "time": "2025-04-01T10:36:49Z"
                },
                {
                    "accel_x": 0.689472,
                    "accel_y": -0.104939,
                    "accel_z": 9.682153,
                    "gyro_x": -0.000083,
                    "gyro_y": -0.000099,
                    "gyro_z": 0.00009,
                    "target": "Normal Walk",
                    "time": "2025-04-01T10:36:49Z"
                },
                {
                    "accel_x": 0.689472,
                    "accel_y": -0.104939,
                    "accel_z": 9.684546,
                    "gyro_x": -0.000083,
                    "gyro_y": -0.000365,
                    "gyro_z": 0.000357,
                    "target": "Normal Walk",
                    "time": "2025-04-01T10:36:49Z"
                },
                {
                    "accel_x": 0.689472,
                    "accel_y": -0.107333,
                    "accel_z": 9.694123,
                    "gyro_x": -0.000083,
                    "gyro_y": -0.000099,
                    "gyro_z": 0.00009,
                    "target": "Normal Walk",
                    "time": "2025-04-01T10:36:49Z"
                },
                {
                    "accel_x": 0.682289,
                    "accel_y": -0.104938,
                    "accel_z": 9.696518,
                    "gyro_x": 0.000184,
                    "gyro_y": 0.000168,
                    "gyro_z": -0.000176,
                    "target": "Normal Walk",
                    "time": "2025-04-01T10:36:49Z"
                },
                {
                    "accel_x": 0.687078,
                    "accel_y": -0.097756,
                    "accel_z": 9.689334,
                    "gyro_x": -0.000083,
                    "gyro_y": 0.000168,
                    "gyro_z": -0.000176,
                    "target": "Normal Walk",
                    "time": "2025-04-01T10:36:49Z"
                }
            ],
            "timestamp": "2025-04-01T10:36:49Z"
        }
    }
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