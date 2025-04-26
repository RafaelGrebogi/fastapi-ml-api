import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = Path("models/model.pkl")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

def run_ml_pipeline(mode: str, data_path: str) -> dict:
    df = pd.read_csv(data_path)

    if mode == "training":
        return train_model(df)
    elif mode == "testing":
        return test_model(df)
    elif mode == "production":
        return predict_model(df)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def train_model(df: pd.DataFrame) -> dict:
    # Drop non-numeric tracking columns
    df = df.drop(columns=[col for col in df.columns if col in ("window_id", "msg_id")], errors="ignore")


    X = df.drop(columns=["label"])
    y = df["label"]


    model = RandomForestClassifier()
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)

    return {
        "status": "training complete",
        "model_path": str(MODEL_PATH),
        # "features": list(X.columns),
        "samples": len(X),
    }

def test_model(df: pd.DataFrame) -> dict:
    X = df.drop(columns=["label"])
    y_true = df["label"]

    if not MODEL_PATH.exists():
        return {"error": "Model not found. Please train first."}

    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X)

    report = classification_report(y_true, y_pred, output_dict=True)
    results_path = RESULTS_DIR / "test_results.json"

    with open(results_path, "w") as f:
        json.dump({
            "metrics": report,
            "predictions": y_pred.tolist(),
            "targets": y_true.tolist()
        }, f, indent=2)

    return {
        "status": "testing complete",
        "results_file": str(results_path),
        "samples": len(X),
    }

def predict_model(df: pd.DataFrame) -> dict:
    if not MODEL_PATH.exists():
        return {"error": "Model not found. Please train first."}

    model = joblib.load(MODEL_PATH)
    predictions = model.predict(df)

    results_path = RESULTS_DIR / "predictions.json"
    with open(results_path, "w") as f:
        json.dump({
            "predictions": predictions.tolist()
        }, f, indent=2)

    return {
        "status": "prediction complete",
        "results_file": str(results_path),
        "samples": len(df),
    }
