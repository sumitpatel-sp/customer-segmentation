from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import os
import pandas as pd

from src.train_ml import train_kmeans
from src.predict import predict_cluster


app = FastAPI()

MODEL_PATH = "models/kmeans.pkl"
SCALER_PATH = "models/scaler.pkl"
DATA_PATH = "data/Mall_Customers.csv"

model = None
scaler = None


def _ensure_models_loaded():
    global model, scaler

    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return

    train_kmeans()
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)


@app.on_event("startup")
def _startup():
    _ensure_models_loaded()


def _prepare_features(df: pd.DataFrame) -> np.ndarray:
    df_proc = df.copy()
    if "CustomerID" in df_proc.columns:
        df_proc = df_proc.drop("CustomerID", axis=1)

    if "Gender" in df_proc.columns:
        df_proc["Gender"] = df_proc["Gender"].map({"Male": 0, "Female": 1})

    features = df_proc[["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]]
    return features.to_numpy()


class Customer(BaseModel):
    Gender: int
    Age: int
    Income: float  # Annual income in k$
    Spending: float  # Spending score (1-100)


@app.post("/predict")
def predict(customer: Customer):
    _ensure_models_loaded()
    data = [
        customer.Gender,
        customer.Age,
        customer.Income,
        customer.Spending,
    ]
    cluster = predict_cluster(data, model=model, scaler=scaler)
    return {"Cluster": cluster}


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
    X = _prepare_features(df)
    X_scaled = scaler.transform(X)
    clusters = model.predict(X_scaled)

    df_with_clusters = df.copy()
    df_with_clusters["Cluster"] = clusters

    cluster_summary = (
        df_with_clusters.groupby("Cluster")[["Annual Income (k$)", "Spending Score (1-100)"]]
        .mean()
        .reset_index()
    )

    segment_sizes = (
        df_with_clusters["Cluster"]
        .value_counts()
        .sort_index()
        .rename_axis("Cluster")
        .reset_index(name="Count")
    )

    meanings = {}
    for _, row in cluster_summary.iterrows():
        cluster_id = int(row["Cluster"])
        income = float(row["Annual Income (k$)"])
        spending = float(row["Spending Score (1-100)"])

        if income < 50 and spending < 50:
            label = "Low Income - Low Spending"
        elif income >= 50 and spending >= 50:
            label = "High Income - High Spending"
        elif income >= 50 and spending < 50:
            label = "High Income - Low Spending"
        else:
            label = "Low Income - High Spending"

        meanings[cluster_id] = label

    return {
        "total_customers": int(len(df_with_clusters)),
        "n_segments": int(df_with_clusters["Cluster"].nunique()),
        "cluster_summary": cluster_summary.to_dict(orient="records"),
        "segment_sizes": segment_sizes.to_dict(orient="records"),
        "segment_meanings": meanings,
    }


@app.post("/retrain")
def retrain_model():
    train_kmeans()
    _ensure_models_loaded()
    return {"message": "Model retrained successfully"}