from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

from src.train_ml import train_kmeans
from src.predict import predict_cluster


app = FastAPI()

# Load trained artefacts once at startup
model = joblib.load("models/kmeans.pkl")
scaler = joblib.load("models/scaler.pkl")


class Customer(BaseModel):
    Gender: int
    Age: int
    Income: float  # Annual income in k$
    Spending: float  # Spending score (1-100)


@app.post("/predict")
def predict(customer: Customer):
    data = [
        customer.Gender,
        customer.Age,
        customer.Income,
        customer.Spending,
    ]
    cluster = predict_cluster(data, model=model, scaler=scaler)
    return {"Cluster": cluster}


@app.post("/retrain")
def retrain_model():
    train_kmeans()
    return {"message": "Model retrained successfully"}