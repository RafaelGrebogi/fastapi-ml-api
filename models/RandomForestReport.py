import json
import pandas as pd
import numpy as np
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# === Config ===
MODEL_PATH = "models/modelv1.pkl"  # your pickled RandomForest model
FEATURE_DATA_PATH = "data/features/C85D60BD9E7C_session1_20250514_191157.csv"  # to extract column names
OUTPUT_PATH = "model_report.json"
TOP_N_FEATURES = 20

# === Load model ===
import joblib
model = joblib.load(MODEL_PATH)

# === Load feature names ===
data = pd.read_csv(FEATURE_DATA_PATH)
feature_names = data.drop(columns=["label"], errors='ignore').columns.tolist()

# === Extract model info ===
model_type = type(model).__name__
n_estimators = len(model.estimators_)
params = model.get_params()

# === Feature importance ===
importances = model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
top_features = [{
    "feature": feature_names[i],
    "importance": float(importances[i])
} for i in sorted_idx[:TOP_N_FEATURES]]

# === Build report ===
report = {
    "model_type": model_type,
    "n_estimators": n_estimators,
    "hyperparameters": params,
    "top_features": top_features,
    "all_importances": {
        feature_names[i]: float(importances[i]) for i in range(len(importances))
    }
}

# === Save JSON ===
with open(OUTPUT_PATH, "w") as f:
    json.dump(report, f, indent=2)

print(f"Model report saved to {OUTPUT_PATH}")
