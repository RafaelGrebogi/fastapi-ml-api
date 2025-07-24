from resultsPlot2db import plot_motion_density, store_supabase_bucket

import json




with open("/mnt/c/Users/rafag/ml-api/data/esp32-v1_TrainData.json", "r") as f:
    all_data = json.load(f)
    # Navigate into the nested 'TrainingDataset' branch
    test_data = all_data.get("ESP32_Develop", {}).get("TrainingDataset", {})


_, image_bytes = plot_motion_density(test_data, window_size=3, bw_adjust=0.6)

filename = "density.png"
mode = "Training"
user_id = 123

success = store_supabase_bucket(image_bytes=image_bytes, filename=filename, mode=mode, user_id=user_id)