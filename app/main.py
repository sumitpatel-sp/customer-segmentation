from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()

model = joblib.load("models/kmeans.pkl")

class Customer(BaseModel):
    Gender: int
    Age: int
    Income: float
    Spending: float

@app.post("/predict")
def predict(customer: Customer):

    data = np.array([[customer.Gender,
                      customer.Age,
                      customer.Income,
                      customer.Spending]])

    cluster = model.predict(data)

    return {"Cluster": int(cluster[0])}
