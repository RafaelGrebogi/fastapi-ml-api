import json
from feature_extraction import extract_features_from_firebase_batch

# Simulated batch data as if received from Firebase
test_data = {
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

# Run the feature extraction
success, output_dir = extract_features_from_firebase_batch(test_data)

if success:
    print(f"\n✅ Test successful! Features saved in: {output_dir}")
else:
    print("\n❌ Test failed.")
