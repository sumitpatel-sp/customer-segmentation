import datetime as dt
import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess(df: pd.DataFrame):
    """
    Clean the UK e-commerce transaction DataFrame, build RFM features,
    remove outliers and scale.

    Returns
    -------
    rfm        : pd.DataFrame  — per-customer RFM table (unscaled, outliers removed)
    rfm_scaled : np.ndarray    — StandardScaler-transformed RFM features
    scaler     : StandardScaler — fitted scaler (save with model)
    """
    # ── 1. Clean ────────────────────────────────────────────────────────────
    df = df.dropna(subset=["CustomerID"])
    df = df[df["Quantity"] > 0]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"]  = df["Quantity"] * df["UnitPrice"]

    # ── 2. RFM aggregation ──────────────────────────────────────────────────
    reference_date = df["InvoiceDate"].max() + dt.timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency   = ("InvoiceDate",  lambda x: (reference_date - x.max()).days),
        Frequency = ("InvoiceNo",    "count"),
        Monetary  = ("TotalPrice",   "sum"),
    )

    # ── 3. Remove outliers (99th percentile) ────────────────────────────────
    rfm = rfm[rfm["Monetary"]  < rfm["Monetary"].quantile(0.99)]
    rfm = rfm[rfm["Frequency"] < rfm["Frequency"].quantile(0.99)]

    # ── 4. Scale ─────────────────────────────────────────────────────────────
    scaler     = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

    return rfm, rfm_scaled, scaler
