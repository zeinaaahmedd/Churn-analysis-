import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict

app = FastAPI(title="Churn Prediction API")

# Load all artifacts once at startup
MODEL          = joblib.load("models/xgboost.pkl")
LABEL_ENCODERS = joblib.load("models/label_encoders.pkl")  # dict: col -> fitted LabelEncoder
SCALER         = joblib.load("models/scaler.pkl")
FEATURE_COLS   = joblib.load("models/feature_columns.pkl") # ordered list matching training columns

# numeric columns are everything not label-encoded
NUMERIC_COLS = [c for c in FEATURE_COLS if c not in LABEL_ENCODERS]


class CustomerRecord(BaseModel):
    # raw customer fields before any encoding or scaling
    features: Dict[str, Any]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(record: CustomerRecord):
    data = record.features

    missing = [c for c in FEATURE_COLS if c not in data]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing fields: {missing}")

    row = pd.DataFrame([{col: data[col] for col in FEATURE_COLS}])

    # apply the same label encoding used during training
    for col, le in LABEL_ENCODERS.items():
        val = str(row.at[0, col])
        if val not in le.classes_:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown value '{val}' for field '{col}'. Valid values: {list(le.classes_)}"
            )
        row[col] = le.transform([val])

    row[NUMERIC_COLS] = SCALER.transform(row[NUMERIC_COLS])

    prob = float(MODEL.predict_proba(row)[0][1])
    pred = int(prob >= 0.5)

    return {"churn_probability": round(prob, 4), "prediction": pred}
