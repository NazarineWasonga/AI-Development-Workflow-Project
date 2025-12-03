from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# Load saved model
model = joblib.load("../model.pkl")
scaler = joblib.load("../scaler.pkl")

@app.post("/predict")
def predict(data: dict):
    """
    Example payload:
    {
        "age": 56,
        "blood_pressure": 130,
        "heart_rate": 88,
        "previous_admissions": 2,
        "diabetes": 1
    }
    """
    values = np.array([list(data.values())]).reshape(1, -1)
    values_scaled = scaler.transform(values)
    
    prediction = model.predict(values_scaled)[0]
    return {"readmission_risk": int(prediction)}
