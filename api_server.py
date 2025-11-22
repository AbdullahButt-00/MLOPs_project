from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
from prometheus_client import Counter, Histogram, make_asgi_app
import time

app = FastAPI(title="Federated Churn Prediction API")

# Prometheus metrics
PREDICTION_COUNTER = Counter('predictions_total', 'Total predictions made')
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency')
CHURN_PREDICTIONS = Counter('churn_predictions', 'Churn predictions', ['label'])

# Load model and preprocessor
model = tf.keras.models.load_model('model/federated_churn_model.h5')
with open('model/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

class CustomerData(BaseModel):
    Tenure: float
    PreferredLoginDevice: str
    CityTier: int
    WarehouseToHome: float
    PreferredPaymentMode: str
    Gender: str
    HourSpendOnApp: float
    NumberOfDeviceRegistered: int
    PreferedOrderCat: str
    SatisfactionScore: int
    MaritalStatus: str
    NumberOfAddress: int
    Complain: int
    OrderAmountHikeFromlastYear: float
    CouponUsed: float
    OrderCount: float
    DaySinceLastOrder: float
    CashbackAmount: float

@app.post("/predict")
async def predict_churn(data: CustomerData):
    start_time = time.time()
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame([data.dict()])
        
        # Preprocess
        X_transformed = preprocessor.transform(df)
        
        # Predict
        prediction_prob = float(model.predict(X_transformed, verbose=0)[0][0])
        prediction_label = int(prediction_prob >= 0.5)
        
        # Update metrics
        PREDICTION_COUNTER.inc()
        CHURN_PREDICTIONS.labels(label=str(prediction_label)).inc()
        
        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)
        
        return {
            "churn_probability": prediction_prob,
            "churn_prediction": prediction_label,
            "risk_level": "High" if prediction_prob > 0.7 else "Medium" if prediction_prob > 0.4 else "Low",
            "latency_ms": latency * 1000
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)