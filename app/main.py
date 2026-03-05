from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import os
import pandas as pd

from src.train_ml import train_kmeans
from src.predict import predict_cluster
from src.segment_logic import get_segment_info


app = FastAPI()

MODEL_PATH  = "models/kmeans.pkl"
SCALER_PATH = "models/scaler.pkl"
DATA_PATH   = "data/Mall_Customers.csv"

model  = None
scaler = None


def _ensure_models_loaded():
    global model, scaler
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model  = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return
    train_kmeans()
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)


@app.on_event("startup")
def _startup():
    _ensure_models_loaded()


class Customer(BaseModel):
    Gender:   int
    Age:      int
    Income:   float   # Annual income in k$
    Spending: float   # Spending score (1-100)


@app.post("/predict")
def predict(customer: Customer):
    """
    1. KMeans runs on all 4 features → returns cluster number
    2. Segment is determined from the customer's own income & spending
       using threshold rules (income >= 50k$ = High Income, spending >= 50 = High Spending)
    3. Returns both cluster number (for reference) and segment name
    """
    _ensure_models_loaded()

    data    = [customer.Gender, customer.Age, customer.Income, customer.Spending]
    cluster = predict_cluster(data, model=model, scaler=scaler)

    # Determine segment from this customer's own income & spending values
    seg_info = get_segment_info(customer.Income, customer.Spending)

    return {
        "Cluster":      cluster,
        "segment_name": seg_info["name"],
        "description":  seg_info["description"],
        "strategy":     seg_info["strategy"],
    }


@app.get("/health")
def health():
    ok = os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)
    return {"status": "ok", "models_present": ok}


@app.get("/segments/summary")
def segment_summary():
    _ensure_models_loaded()

    if not os.path.exists(DATA_PATH):
        return {"error": f"Dataset not found at {DATA_PATH}"}

    df = pd.read_csv(DATA_PATH)
    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
    X = df[["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]].to_numpy()

    df["Cluster"] = model.predict(scaler.transform(X))
    cluster_summary = (
        df.groupby("Cluster")[["Annual Income (k$)", "Spending Score (1-100)"]]
        .mean().reset_index()
    )
    segment_sizes = (
        df["Cluster"].value_counts().sort_index()
        .rename_axis("Cluster").reset_index(name="Count")
    )
    scatter_data = df[["Annual Income (k$)", "Spending Score (1-100)", "Cluster"]].to_dict(orient="records")

    meanings = descriptions = strategies = {}
    for _, row in cluster_summary.iterrows():
        cid  = int(row["Cluster"])
        info = get_segment_info(float(row["Annual Income (k$)"]), float(row["Spending Score (1-100)"]))
        meanings[cid]     = info["name"]
        descriptions[cid] = info["description"]
        strategies[cid]   = info["strategy"]

    return {
        "total_customers":      int(len(df)),
        "n_segments":           int(df["Cluster"].nunique()),
        "cluster_summary":      cluster_summary.to_dict(orient="records"),
        "segment_sizes":        segment_sizes.to_dict(orient="records"),
        "segment_meanings":     meanings,
        "segment_descriptions": descriptions,
        "segment_strategies":   strategies,
        "scatter_data":         scatter_data,
    }


@app.post("/retrain")
def retrain_model():
    train_kmeans()
    _ensure_models_loaded()
    return {"message": "Model retrained successfully"}