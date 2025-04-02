# backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import logging
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class FinancialData(BaseModel):
    income: float
    expenses: float
    debt: float  # Ensure debt is included
    

class PredictionResponse(BaseModel):
    prediction: float
    risk_score: float
    savings_rate: float
    recommendations: List[str]

# Load artifacts at startup
model = None
scaler = None

@app.on_event("startup")
async def load_artifacts():
    global model, scaler
    try:
        model = joblib.load('regression_model.pkl')
        scaler = joblib.load('scaler.pkl')
        logging.info("Successfully loaded model and scaler")
    except Exception as e:
        logging.error(f"Failed to load artifacts: {str(e)}")
        raise RuntimeError("Model loading failed")

def prepare_features(data: FinancialData) -> np.ndarray:
    """Process a single entry (no lag features)."""
    try:
        # Calculate savings rate
        savings_rate = (data.income - data.expenses) / data.income
        
        # Create feature array (match what your model expects)
        features = np.array([
            [savings_rate]
        ])
        
        return features
    except Exception as e:
        logging.error(f"Feature preparation failed: {str(e)}")
        raise

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: FinancialData):  # Accepts a single object, not a list!
    try:
        # Prepare features
        features = prepare_features(data)
        
        # Scale features
        scaled_features = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]

        # Generate recommendations
        recommendations = []
        debt_ratio = data.debt / data.income
        savings_rate = (data.income - data.expenses) / data.income
        
        if debt_ratio > 0.35:
            recommendations.append("Reduce debt exposure")
        if savings_rate < 0.2:
            recommendations.append("Increase savings rate")

        return {
            "prediction": float(prediction),
            "risk_score": debt_ratio * 100,
            "savings_rate": savings_rate * 100,
            "recommendations": recommendations
        }

    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)