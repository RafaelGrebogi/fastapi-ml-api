import json
from feature_extraction import extract_features_from_firebase_batch
from ml_manager import run_ml_pipeline




# Choose which module to test/debug
test_opt = "testing"

# Simulated batch data as if received from Firebase
data_opt = "multi"

if data_opt == "single":
    test_data = {
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
elif data_opt == "multi":
        import json
        nSamples = 10
        # Load the JSON file
        if test_opt == "training":
            with open("/mnt/c/Users/rafag/ml-api/data/esp32-v1_TrainData.json", "r") as f:
                all_data = json.load(f)
        elif test_opt == "testing":
            with open("/mnt/c/Users/rafag/ml-api/data/esp32-v1_TestData.json", "r") as f:
                all_data = json.load(f)
        else:
            with open("/mnt/c/Users/rafag/ml-api/data/esp32-rtdb-ML_TestData.json", "r") as f:
            # with open("/mnt/c/Users/rafag/ml-api/data/esp32-rtdb-Testing.json", "r") as f:
              all_data = json.load(f)

        
        # with open("/mnt/c/Users/rafag/ml-api/data/esp32-v1_TestData.json", "r") as f:
        # with open("/mnt/c/Users/rafag/ml-api/data/esp32-rtdb-ML_TestData.json", "r") as f:
        # with open("/mnt/c/Users/rafag/ml-api/data/esp32-rtdb-Testing.json", "r") as f:
            # all_data = json.load(f)

        # Navigate into the nested 'TrainingDataset' branch
        training_data = all_data.get("ESP32_Develop", {}).get("TrainingDataset", {})

        # Filter for messages with exactly 'nSamples' samples
        test_data = {
            msg_id: msg
            for msg_id, msg in training_data.items()
            if len(msg.get("samples", [])) == nSamples
        }

        # Check how many were added
        print(f"{len(test_data)} messages found.")


if test_opt == "feature":
    # Run the feature extraction
    success, output_dir = extract_features_from_firebase_batch(test_data,  USE_MULTI_MESSAGE_WINDOW=True)

    if success:
        print(f"\n✅ Test successful! Features saved in: {output_dir}")
    else:
        print("\n❌ Test failed.")

elif test_opt == "training":
     # Run the feature extraction
    success, output_dir = extract_features_from_firebase_batch(test_data,  USE_MULTI_MESSAGE_WINDOW=True)

    if success:
        print(f"\n✅ Test successful! Features saved in: {output_dir}")
    else:
        print("\n❌ Test failed.")
    # Run testing pipeline
    result = run_ml_pipeline("training", output_dir)
    print("✅ ML Pipeline Result:", result)

elif test_opt == "testing":
    # Run the feature extraction
    success, output_dir = extract_features_from_firebase_batch(test_data,  USE_MULTI_MESSAGE_WINDOW=True)

    if success:
        print(f"\n✅ Test successful! Features saved in: {output_dir}")
    else:
        print("\n❌ Test failed.")
    # Run testing pipeline
    result = run_ml_pipeline("testing", output_dir)
    print("✅ ML Pipeline Result:", result)

elif test_opt == "production":  
    # Run the feature extraction
    success, output_dir = extract_features_from_firebase_batch(test_data,  USE_MULTI_MESSAGE_WINDOW=True)

    if success:
        print(f"\n✅ Test successful! Features saved in: {output_dir}")
    else:
        print("\n❌ Test failed.")
    # Run testing pipeline
    result = run_ml_pipeline("production", output_dir)
    print("✅ ML Pipeline Result:", result)






