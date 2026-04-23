from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import sys
sys.path.append(r"C:\Users\dipin\fraud-detection-system")

from src.api.predict import predict_fraud, THRESHOLD

# Create the FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time transaction fraud detection using XGBoost",
    version="1.0.0"
)

# Define what a transaction looks like
# Optional = field is not required (many fields can be missing)
class Transaction(BaseModel):
    TransactionAmt: float
    ProductCD: Optional[str] = None
    card1: Optional[float] = None
    card2: Optional[float] = None
    card3: Optional[float] = None
    card4: Optional[str] = None
    card5: Optional[float] = None
    card6: Optional[str] = None
    addr1: Optional[float] = None
    addr2: Optional[float] = None
    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None
    TransactionDT: Optional[float] = None

# Endpoint 1: Health check
@app.get("/")
def root():
    return {
        "status": "online",
        "message": "Fraud Detection API is running"
    }

# Endpoint 2: Model info
@app.get("/model-info")
def model_info():
    return {
        "model": "XGBoost (Optuna tuned)",
        "auc_roc": 0.9566,
        "f1_score": 0.7020,
        "precision": 0.7803,
        "recall": 0.6381,
        "threshold": THRESHOLD,
        "trained_on": "IEEE-CIS Fraud Dataset (590K transactions)"
    }

# Endpoint 3: Predict fraud
@app.post("/predict")
def predict(transaction: Transaction):
    # Convert to dict, drop None values
    data = {k: v for k, v in transaction.dict().items() if v is not None}
    result = predict_fraud(data)
    return result
