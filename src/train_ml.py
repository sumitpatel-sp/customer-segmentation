import os
import json
import joblib
import pandas as pd
from sklearn.cluster import KMeans

from src.data_loader import load_data
from src.preprocessing import preprocess
from src.segment_logic import label_segment


def _build_cluster_map(model, scaler, rfm: pd.DataFrame) -> dict:
    """
    Map each KMeans cluster ID → segment label using a rank-based approach.
    This guarantees all 4 segments are always distinct regardless of centroid values.

    Ranking logic (applied to centroid means):
    - "High Value" → highest composite score  (high Monetary + high Frequency + low Recency)
    - "Loyal"      → second highest Frequency (but not the top composite)
    - "At Risk"    → highest Recency          (least recent buyer)
    - "Potential Loyalists"  → remaining cluster
    """
    rfm = rfm.copy()
    rfm["Cluster"] = model.predict(scaler.transform(rfm[["Recency", "Frequency", "Monetary"]]))

    c = (
        rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]]
        .mean()
        .reset_index()
    )

    # Normalise each column to 0-1 so they're comparable
    for col in ["Recency", "Frequency", "Monetary"]:
        col_range = c[col].max() - c[col].min()
        if col_range > 0:
            c[col + "_n"] = (c[col] - c[col].min()) / col_range
        else:
            c[col + "_n"] = 0.5

    # Composite value score: high Monetary + high Frequency + low Recency
    c["value_score"] = c["Monetary_n"] + c["Frequency_n"] - c["Recency_n"]

    # Assign segments by rank — each cluster gets exactly one unique label
    cluster_map = {}

    # 1. High Value  → best composite score
    hv_idx = int(c.loc[c["value_score"].idxmax(), "Cluster"])
    cluster_map[hv_idx] = "High Value"
    remaining = c[c["Cluster"] != hv_idx]

    # 2. At Risk     → highest Recency in the remaining clusters
    ar_idx = int(remaining.loc[remaining["Recency"].idxmax(), "Cluster"])
    cluster_map[ar_idx] = "At Risk"
    remaining = remaining[remaining["Cluster"] != ar_idx]

    # 3. Loyal       → next highest Frequency in the remaining clusters
    lo_idx = int(remaining.loc[remaining["Frequency"].idxmax(), "Cluster"])
    cluster_map[lo_idx] = "Loyal"
    remaining = remaining[remaining["Cluster"] != lo_idx]

    # 4. Low Value   → whatever is left
    lv_idx = int(remaining.iloc[0]["Cluster"])
    cluster_map[lv_idx] = "Potential Loyalists"

    return cluster_map


def train_kmeans():
    # ── Load & preprocess ───────────────────────────────────────────────────
    df = load_data("data/data.csv", encoding="ISO-8859-1")
    rfm, rfm_scaled, scaler = preprocess(df)

    print(f"RFM shape after cleaning: {rfm.shape}")

    # ── Train KMeans ────────────────────────────────────────────────────────
    model = KMeans(n_clusters=4, random_state=42, n_init=10)
    model.fit(rfm_scaled)

    kmeans_labels       = model.predict(rfm_scaled)
    rfm["KMeans_Cluster"] = kmeans_labels

    # ── Segment labels (rule-based on raw RFM values) ───────────────────────
    rfm["Segment"] = rfm.apply(
        lambda row: label_segment(row["Recency"], row["Frequency"], row["Monetary"]),
        axis=1,
    )
    print("Segment distribution:")
    print(rfm["Segment"].value_counts())

    # ── Save artifacts ──────────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    joblib.dump(model,  "models/kmeans.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    cluster_map = _build_cluster_map(model, scaler, rfm)
    with open("models/cluster_map.json", "w") as f:
        json.dump(cluster_map, f, indent=2)

    rfm.to_csv("rfm_segments.csv")

    print("✅ Model, scaler, cluster_map and rfm_segments.csv saved.")
    print(f"   Cluster map: {cluster_map}")
    return rfm


if __name__ == "__main__":
    train_kmeans()
