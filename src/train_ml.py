import os
import json
import joblib
import numpy as np
from sklearn.cluster import KMeans
from src.data_loader import load_data
from src.preprocessing import preprocess
from src.segment_logic import get_segment_info


def _build_cluster_map(model, scaler, df):
    """
    Assign a unique segment label to each KMeans cluster by inspecting
    cluster centroids' average Annual Income and Spending Score.
    """
    import pandas as pd

    df = df.copy()
    X = df[["Annual Income (k$)", "Spending Score (1-100)"]].to_numpy()
    df["Cluster"] = model.predict(scaler.transform(X))

    # Mean income & spending per cluster (in original un-scaled units)
    centroid_stats = (
        df.groupby("Cluster")[["Annual Income (k$)", "Spending Score (1-100)"]]
        .mean()
        .reset_index()
    )

    half = len(centroid_stats) // 2
    inc_ranks = centroid_stats["Annual Income (k$)"].rank(ascending=False, method="first").astype(int)
    high_inc_idx = centroid_stats.index[inc_ranks <= half].tolist()
    low_inc_idx  = centroid_stats.index[inc_ranks >  half].tolist()

    hi_grp = centroid_stats.loc[high_inc_idx].sort_values("Spending Score (1-100)", ascending=False)
    lo_grp = centroid_stats.loc[low_inc_idx ].sort_values("Spending Score (1-100)", ascending=False)

    cluster_map = {}
    if len(hi_grp) >= 1:
        cluster_map[int(hi_grp.iloc[0]["Cluster"])] = "High Income - High Spending"
    if len(hi_grp) >= 2:
        cluster_map[int(hi_grp.iloc[1]["Cluster"])] = "High Income - Low Spending"
    if len(lo_grp) >= 1:
        cluster_map[int(lo_grp.iloc[0]["Cluster"])] = "Low Income - High Spending"
    if len(lo_grp) >= 2:
        cluster_map[int(lo_grp.iloc[1]["Cluster"])] = "Low Income - Low Spending"

    return cluster_map


def train_kmeans():

    # Load dataset
    df = load_data("data/Mall_Customers.csv")

    # Preprocess
    data, scaler = preprocess(df)

    # Train model
    model = KMeans(n_clusters=4, random_state=42)
    model.fit(data)

    # Create models folder
    os.makedirs("models", exist_ok=True)

    # Save model and scaler
    joblib.dump(model, "models/kmeans.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    # Build and save cluster → segment mapping
    cluster_map = _build_cluster_map(model, scaler, df)
    with open("models/cluster_map.json", "w") as f:
        json.dump(cluster_map, f, indent=2)

    print("✅ Model, scaler, and cluster_map saved successfully")
    print(f"   Cluster map: {cluster_map}")


if __name__ == "__main__":
    train_kmeans()
