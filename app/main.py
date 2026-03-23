from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import os
import json
import pandas as pd

from src.train_ml import train_kmeans
from src.predict import predict_cluster
from src.segment_logic import get_segment_info_by_name


app = FastAPI()

MODEL_PATH       = "models/kmeans.pkl"
SCALER_PATH      = "models/scaler.pkl"
CLUSTER_MAP_PATH = "models/cluster_map.json"
RFM_PATH         = "rfm_segments.csv"

model       = None
scaler      = None
cluster_map = {}   # {cluster_id (int): segment_name (str)}


def _ensure_models_loaded():
    global model, scaler, cluster_map
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model  = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    else:
        train_kmeans()
        model  = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

    if os.path.exists(CLUSTER_MAP_PATH):
        with open(CLUSTER_MAP_PATH) as f:
            raw = json.load(f)
        cluster_map = {int(k): v for k, v in raw.items()}
    else:
        train_kmeans()
        model  = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        with open(CLUSTER_MAP_PATH) as f:
            raw = json.load(f)
        cluster_map = {int(k): v for k, v in raw.items()}


@app.on_event("startup")
def _startup():
    _ensure_models_loaded()


class Customer(BaseModel):
    Recency:   int    # days since last purchase
    Frequency: int    # number of orders
    Monetary:  float  # total spend (£)


@app.post("/predict")
def predict(customer: Customer):
    """
    Predict RFM-based customer segment.
    Returns cluster number, segment name, description, strategy and goal.
    """
    _ensure_models_loaded()

    data    = [customer.Recency, customer.Frequency, customer.Monetary]
    cluster = predict_cluster(data, model=model, scaler=scaler)

    seg_name = cluster_map.get(cluster, "Low Value")
    seg_info = get_segment_info_by_name(seg_name)

    return {
        "Cluster":      cluster,
        "segment_name": seg_name,
        "description":  seg_info.get("description", ""),
        "strategy":     seg_info.get("strategy", ""),
        "goal":         seg_info.get("goal", ""),
    }


@app.get("/health")
def health():
    ok = (
        os.path.exists(MODEL_PATH)
        and os.path.exists(SCALER_PATH)
        and os.path.exists(CLUSTER_MAP_PATH)
    )
    return {"status": "ok", "models_present": ok}


@app.get("/cluster-map")
def get_cluster_map():
    _ensure_models_loaded()
    return {"cluster_map": cluster_map}


@app.get("/segments/summary")
def segment_summary():
    _ensure_models_loaded()

    if not os.path.exists(RFM_PATH):
        return {"error": f"RFM dataset not found at {RFM_PATH}. Run training first."}

    rfm = pd.read_csv(RFM_PATH, index_col=0)

    cluster_prof = (
        rfm.groupby("KMeans_Cluster")[["Recency", "Frequency", "Monetary"]]
        .mean()
        .reset_index()
    )
    segment_sizes = (
        rfm["Segment"].value_counts()
        .rename_axis("Segment")
        .reset_index(name="Count")
    )

    meanings     = {cid: cluster_map.get(cid, "Unknown") for cid in cluster_map}
    descriptions = {cid: get_segment_info_by_name(seg).get("description", "") for cid, seg in cluster_map.items()}
    strategies   = {cid: get_segment_info_by_name(seg).get("strategy", "")    for cid, seg in cluster_map.items()}

    return {
        "total_customers":       int(len(rfm)),
        "n_segments":            int(rfm["Segment"].nunique()),
        "cluster_summary":       cluster_prof.to_dict(orient="records"),
        "segment_sizes":         segment_sizes.to_dict(orient="records"),
        "segment_meanings":      meanings,
        "segment_descriptions":  descriptions,
        "segment_strategies":    strategies,
        "cluster_map":           cluster_map,
    }


@app.post("/retrain")
def retrain_model():
    global cluster_map
    train_kmeans()
    _ensure_models_loaded()
    return {"message": "Model retrained successfully", "cluster_map": cluster_map}
