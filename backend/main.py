from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = joblib.load("cardio_pipeline.pkl")

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
