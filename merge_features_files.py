import pandas as pd
import glob
import os

def merge_feature_csvs(input_dir="data/features", output_file="data/datasets/training_dataset.csv"):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Get new CSVs from feature folder
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        print(" No feature CSV files found.")
        return False

    new_data = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    # Load existing dataset if it exists
    if os.path.exists(output_file):
        existing_data = pd.read_csv(output_file)
        combined = pd.concat([existing_data, new_data], ignore_index=True)

        # Remove duplicates based on 'msg_id'
        if 'msg_id' in combined.columns:
            combined.drop_duplicates(subset='msg_id', inplace=True)
        else:
            print("⚠️ 'msg_id' not found — cannot deduplicate.")
    else:
        combined = new_data

    combined.to_csv(output_file, index=False)
    print(f"✅ Appended data saved to: {output_file} ({len(combined)} total rows)")
    return True

# Run if executed directly
if __name__ == "__main__":
    merge_feature_csvs()
