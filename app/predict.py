# app/predict.py
from joblib import load
import pandas as pd
from typing import Dict, Any

MODEL_PATH = "/workspace/app/model.joblib"
model = load(MODEL_PATH)

CAT_COLS = ["fuel", "seller_type", "transmission", "brand"]
NUM_COLS = ["year", "km_driven", "owner", "engine", "max_power", "mileage", "seats"]
ALL_COLS = CAT_COLS + NUM_COLS

def predict_price(payload: Dict[str, Any]) -> float:
    row = {c: None for c in ALL_COLS}
    row.update(payload)
    X = pd.DataFrame([row], columns=ALL_COLS)
    return float(model.predict(X)[0])