import joblib
import numpy as np
import pandas as pd
import json
from pathlib import Path

MODEL_PATH   = Path(r"C:\Users\dipin\fraud-detection-system\models\xgb_tuned.pkl")
FEATURES_PATH = Path(r"C:\Users\dipin\fraud-detection-system\models\feature_names.json")

model = joblib.load(MODEL_PATH)
with open(FEATURES_PATH) as f:
    FEATURE_NAMES = json.load(f)

THRESHOLD = 0.5905
CAT_COLUMNS = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain"]

def predict_fraud(transaction: dict) -> dict:
    # Start with all features set to 0
    df = pd.DataFrame([{col: 0 for col in FEATURE_NAMES}])

    # Fill in provided values
    for key, val in transaction.items():
        if key in df.columns:
            df[key] = val

    # Encode any categorical columns that were provided
    for col in CAT_COLUMNS:
        if col in transaction and col in df.columns:
            df[col] = pd.Categorical([transaction[col]]).codes[0]

    fraud_probability = model.predict_proba(df)[0][1]
    is_fraud = bool(fraud_probability >= THRESHOLD)

    return {
        "fraud_probability": round(float(fraud_probability), 4),
        "is_fraud": is_fraud,
        "risk_level": "HIGH" if fraud_probability >= 0.7
                      else "MEDIUM" if fraud_probability >= 0.4
                      else "LOW"
    }
