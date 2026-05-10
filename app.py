
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import pandas as pd
import joblib

model = joblib.load("diabetes_model.pkl")
training_columns = joblib.load("training_columns.pkl")
label_encoder = joblib.load("label_encoder.pkl")

app = FastAPI(title="Diabetes Prediction API")

class PatientData(BaseModel):
    age: float
    urea: float
    cr: float
    hba1c: float
    chol: float
    tg: float
    hdl: float
    ldl: float
    vldl: float
    bmi: float
    gender: str

    @validator("gender")
    def validate_gender(cls, v):
        v = v.strip().upper()
        if v not in ["M", "F"]:
            raise ValueError("Gender must be M or F")
        return v

@app.get("/")
def health_check():
    return {"status": "API is running"}

@app.post("/predict")
def predict(patient: PatientData):
    input_data = {
        "AGE": patient.age, "Urea": patient.urea,
        "Cr": patient.cr, "HbA1c": patient.hba1c,
        "Chol": patient.chol, "TG": patient.tg,
        "HDL": patient.hdl, "LDL": patient.ldl,
        "VLDL": patient.vldl, "BMI": patient.bmi,
        "Gender_F": 1.0 if patient.gender == "F" else 0.0,
        "Gender_M": 1.0 if patient.gender == "M" else 0.0,
    }
    input_df = pd.DataFrame([input_data])[training_columns]
    pred = model.predict(input_df)
    label = label_encoder.inverse_transform(pred)[0]
    result = {"Y": "Diabetic", "N": "Non-Diabetic", "P": "Pre-Diabetic"}
    return {"prediction": result.get(label, label), "class_code": label}
