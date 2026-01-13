from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
from pathlib import Path
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
BASE_DIR = Path(__file__).resolve().parent
pipeline = joblib.load(BASE_DIR / "cardio_pipeline.pkl")

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])

    # semantic preprocessing (ONLY this)
    df["age_years"] = df["age"] / 365.25
    df.drop(columns=["age"], inplace=True)

    prob = pipeline.predict_proba(df)[0][1]

    return {
        "probability": float(prob),
        "risk": int(prob >= 0.5)
    }
