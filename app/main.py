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

# Built at startup: maps cluster ID → segment info dict
# e.g. { 0: {"name": "Low Income - Low Spending", ...}, 2: {"name": "High Income - High Spending", ...} }
CLUSTER_TO_SEGMENT = {}

# All segment definitions — used by get_segment_info_by_name
SEGMENT_INFO = {
    "High Income - High Spending": {
        "name":        "High Income - High Spending",
        "description": "Valuable customers already spending a lot",
        "strategy":    "Loyalty rewards, premium membership, early product access",
    },
    "High Income - Low Spending": {
        "name":        "High Income - Low Spending",
        "description": "Have money but are not spending much",
        "strategy":    "Upsell, targeted marketing, premium recommendations",
    },
    "Low Income - High Spending": {
        "name":        "Low Income - High Spending",
        "description": "Spending a lot but are budget sensitive",
        "strategy":    "Discounts, bundles, retarget offers",
    },
    "Low Income - Low Spending": {
        "name":        "Low Income - Low Spending",
        "description": "Low value customers — minimal engagement",
        "strategy":    "Low cost campaigns, awareness campaigns (avoid heavy spend)",
    },
}


def get_segment_info_by_name(name: str) -> dict:
    """Return segment info dict by segment name. Falls back to income/spending rules."""
    return SEGMENT_INFO.get(name, SEGMENT_INFO["Low Income - Low Spending"])



def _build_cluster_segment_map():
    """
    Assign a UNIQUE segment to each cluster by ranking the model's own
    cluster centroids (model.cluster_centers_).

    Uses the model directly — NO CSV file needed — so this always works
    on Render even if data/ is not present.

    Ranking method:
      1. Rank all clusters by their centroid income (feature index 2)
         → top half = "High Income", bottom half = "Low Income"
      2. Within each income group, rank by centroid spending (feature index 3)
         → higher spending = "High Spending"
    """
    global CLUSTER_TO_SEGMENT
    CLUSTER_TO_SEGMENT = {}

    if model is None:
        return

    # model.cluster_centers_ shape: (n_clusters, n_features)
    # Features order: [GenderEncoded, Age, Annual Income (k$), Spending Score (1-100)]
    # Unscale centroids back to original feature space
    try:
        centers_scaled = model.cluster_centers_                    # shape (4, 4)
        centers_orig   = scaler.inverse_transform(centers_scaled)  # back to k$ / score
    except Exception as e:
        print(f"⚠️ Could not unscale centroids: {e}")
        return

    n_clusters = len(centers_orig)
    half       = n_clusters // 2

    # Build a small dataframe of (cluster_id, avg_income, avg_spending)
    rows = []
    for cid, center in enumerate(centers_orig):
        rows.append({
            "Cluster":                cid,
            "Annual Income (k$)":     center[2],   # index 2
            "Spending Score (1-100)": center[3],   # index 3
        })
    summary = pd.DataFrame(rows)

    # Rank by income descending: top half → High Income
    summary["inc_rank"] = summary["Annual Income (k$)"].rank(ascending=False, method="first").astype(int)
    high_inc = summary[summary["inc_rank"] <= half].sort_values("Spending Score (1-100)", ascending=False)
    low_inc  = summary[summary["inc_rank"] >  half].sort_values("Spending Score (1-100)", ascending=False)

    seg_assignments = {}
    if len(high_inc) >= 1: seg_assignments[int(high_inc.iloc[0]["Cluster"])] = "High Income - High Spending"
    if len(high_inc) >= 2: seg_assignments[int(high_inc.iloc[1]["Cluster"])] = "High Income - Low Spending"
    if len(low_inc)  >= 1: seg_assignments[int(low_inc.iloc[0]["Cluster"])]  = "Low Income - High Spending"
    if len(low_inc)  >= 2: seg_assignments[int(low_inc.iloc[1]["Cluster"])]  = "Low Income - Low Spending"

    for cluster_id, seg_name in seg_assignments.items():
        CLUSTER_TO_SEGMENT[cluster_id] = get_segment_info_by_name(seg_name)

    print("Cluster to Segment map built:")
    for cid, info in CLUSTER_TO_SEGMENT.items():
        print(f"  Cluster {cid} -> {info['name']}")


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
    _build_cluster_segment_map()   # build the map right after model loads


def _prepare_features(df: pd.DataFrame) -> np.ndarray:
    df_proc = df.copy()
    if "CustomerID" in df_proc.columns:
        df_proc = df_proc.drop("CustomerID", axis=1)
    if "Gender" in df_proc.columns:
        df_proc["Gender"] = df_proc["Gender"].map({"Male": 0, "Female": 1})
    features = df_proc[["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]]
    return features.to_numpy()


class Customer(BaseModel):
    Gender:   int
    Age:      int
    Income:   float   # Annual income in k$
    Spending: float   # Spending score (1-100)


@app.post("/predict")
def predict(customer: Customer):
    _ensure_models_loaded()
    if not CLUSTER_TO_SEGMENT:
        _build_cluster_segment_map()

    data = [customer.Gender, customer.Age, customer.Income, customer.Spending]

    # Step 1: KMeans assigns a cluster number based on ALL 4 features
    cluster = predict_cluster(data, model=model, scaler=scaler)

    # Step 2: Look up the segment for that cluster from the centroid-based map
    #         This is what gives the ML model its meaning — the cluster number
    #         is now the actual source of truth for the segment label.
    seg_info = CLUSTER_TO_SEGMENT.get(cluster, get_segment_info(customer.Income, customer.Spending))

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
    if not CLUSTER_TO_SEGMENT:
        _build_cluster_segment_map()

    if not os.path.exists(DATA_PATH):
        return {"error": f"Dataset not found at {DATA_PATH}"}

    df = pd.read_csv(DATA_PATH)
    X  = _prepare_features(df)
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

    scatter_data = df_with_clusters[
        ["Annual Income (k$)", "Spending Score (1-100)", "Cluster"]
    ].to_dict(orient="records")

    # Use centroid-based segment map (consistent with /predict)
    meanings     = {}
    strategies   = {}
    descriptions = {}
    for cid, seg_info in CLUSTER_TO_SEGMENT.items():
        meanings[cid]     = seg_info["name"]
        strategies[cid]   = seg_info["strategy"]
        descriptions[cid] = seg_info["description"]

    return {
        "total_customers":   int(len(df_with_clusters)),
        "n_segments":        int(df_with_clusters["Cluster"].nunique()),
        "cluster_summary":   cluster_summary.to_dict(orient="records"),
        "segment_sizes":     segment_sizes.to_dict(orient="records"),
        "segment_meanings":  meanings,
        "segment_strategies": strategies,
        "segment_descriptions": descriptions,
        "scatter_data":      scatter_data,
    }


@app.post("/retrain")
def retrain_model():
    train_kmeans()
    _ensure_models_loaded()
    _build_cluster_segment_map()   # rebuild map after retraining
    return {"message": "Model retrained successfully"}