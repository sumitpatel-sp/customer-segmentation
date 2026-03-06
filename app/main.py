from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import os
import json
import pandas as pd

from src.train_ml import train_kmeans
from src.predict import predict_cluster
from src.segment_logic import get_segment_info


app = FastAPI()

MODEL_PATH       = "models/kmeans.pkl"
SCALER_PATH      = "models/scaler.pkl"
CLUSTER_MAP_PATH = "models/cluster_map.json"
DATA_PATH        = "data/Mall_Customers.csv"

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

    # Load cluster → segment mapping (retrain if missing)
    if os.path.exists(CLUSTER_MAP_PATH):
        with open(CLUSTER_MAP_PATH) as f:
            raw = json.load(f)
        cluster_map = {int(k): v for k, v in raw.items()}
    else:
        # Cluster map missing — retrain to generate it
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
    Gender:   int
    Age:      int
    Income:   float   # Annual income in k$
    Spending: float   # Spending score (1-100)


@app.post("/predict")
def predict(customer: Customer):
    """
    1. KMeans runs on all 4 features → returns cluster number
    2. Segment is determined from the saved cluster_map (built at training time
       by inspecting each cluster's centroid income & spending position).
    3. Returns cluster number, segment name, description, and strategy.
    """
    _ensure_models_loaded()

    data    = [customer.Gender, customer.Age, customer.Income, customer.Spending]
    cluster = predict_cluster(data, model=model, scaler=scaler)

    # ── Segment comes from the cluster map, NOT from threshold rules ──
    seg_name = cluster_map.get(cluster, "Unknown")
    seg_info = get_segment_info_by_name(seg_name)

    return {
        "Cluster":      cluster,
        "segment_name": seg_name,
        "description":  seg_info["description"],
        "strategy":     seg_info["strategy"],
    }


def get_segment_info_by_name(name: str) -> dict:
    """Return description & strategy for a segment name."""
    SEGMENT_DETAILS = {
        "High Income - High Spending": {
            "description": "Valuable customers already spending a lot",
            "strategy":    "Loyalty rewards, premium membership, early product access",
        },
        "High Income - Low Spending": {
            "description": "Have money but not spending much",
            "strategy":    "Upsell, targeted marketing, premium recommendations",
        },
        "Low Income - High Spending": {
            "description": "Spending a lot but budget sensitive",
            "strategy":    "Discounts, bundles, retarget offers",
        },
        "Low Income - Low Spending": {
            "description": "Low value customers",
            "strategy":    "Low cost campaigns, awareness campaigns",
        },
    }
    return SEGMENT_DETAILS.get(name, {"description": "", "strategy": ""})


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
    """Return the cluster → segment mapping for inspection."""
    _ensure_models_loaded()
    return {"cluster_map": cluster_map}


@app.get("/segments/summary")
def segment_summary():
    _ensure_models_loaded()

    if not os.path.exists(DATA_PATH):
        return {"error": f"Dataset not found at {DATA_PATH}"}

    df = pd.read_csv(DATA_PATH)
    X = df[["Annual Income (k$)", "Spending Score (1-100)"]].to_numpy()

    df["Cluster"] = model.predict(scaler.transform(X))

    # Segment is now cluster-driven
    df["Segment"] = df["Cluster"].map(cluster_map)

    cluster_summary = (
        df.groupby("Cluster")[["Annual Income (k$)", "Spending Score (1-100)"]]
        .mean().reset_index()
    )
    segment_sizes = (
        df["Cluster"].value_counts().sort_index()
        .rename_axis("Cluster").reset_index(name="Count")
    )
    scatter_data = df[["Annual Income (k$)", "Spending Score (1-100)", "Cluster", "Segment"]].to_dict(orient="records")

    # Build label dicts from cluster_map
    meanings      = {cid: cluster_map.get(cid, "Unknown") for cid in cluster_map}
    descriptions  = {cid: get_segment_info_by_name(seg)["description"] for cid, seg in cluster_map.items()}
    strategies    = {cid: get_segment_info_by_name(seg)["strategy"]    for cid, seg in cluster_map.items()}

    return {
        "total_customers":       int(len(df)),
        "n_segments":            int(df["Cluster"].nunique()),
        "cluster_summary":       cluster_summary.to_dict(orient="records"),
        "segment_sizes":         segment_sizes.to_dict(orient="records"),
        "segment_meanings":      meanings,
        "segment_descriptions":  descriptions,
        "segment_strategies":    strategies,
        "scatter_data":          scatter_data,
        "cluster_map":           cluster_map,
    }


@app.post("/retrain")
def retrain_model():
    global cluster_map
    train_kmeans()
    _ensure_models_loaded()
    return {"message": "Model retrained successfully", "cluster_map": cluster_map}
