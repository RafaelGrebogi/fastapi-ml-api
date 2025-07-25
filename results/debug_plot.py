from resultsPlot2db import plot_motion_density, store_supabase_bucket
import json
import sys
import os

# # Add the root folder to sys.path so you can import supabase_client
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from context_vars import current_user_id, current_service_id, current_result_id


from dotenv import load_dotenv
load_dotenv()  # Looks for .env file in the current folder



with open("/mnt/c/Users/rafag/ml-api/data/esp32-v1_TrainData.json", "r") as f:
    all_data = json.load(f)
    # Navigate into the nested 'TrainingDataset' branch
    test_data = all_data.get("ESP32_Develop", {}).get("TrainingDataset", {})


_, image_bytes = plot_motion_density(test_data, window_size=3, bw_adjust=0.6)

filename = "density2.png"
mode = "Training"
user_id = 123
service_id = 456
result_id = 789

current_user_id.set(user_id)
current_service_id.set(service_id)
current_result_id.set(result_id)

success = store_supabase_bucket(image_bytes=image_bytes, filename=filename, mode=mode)

if success:
    print(" File saved in Supabase bucket.")
else:
    print(" Failed to save in Supabase bucket.")