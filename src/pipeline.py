import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models"

kmeans      = joblib.load(MODEL_DIR / "kmeans_k4.pkl")
scaler      = joblib.load(MODEL_DIR / "scaler.pkl")

with open(MODEL_DIR / "cluster_label_map.json") as f:
    raw = json.load(f)
    cluster_label_map = {int(k): v for k, v in raw.items()}


def predict_segments(transactions_df, snapshot_date=None):
    """
    Takes a raw transactions dataframe and returns RFM + cluster labels.

    Args:
        transactions_df: DataFrame with columns
                         [CustomerID, InvoiceNo, InvoiceDate, Revenue]
        snapshot_date:   Reference date for recency calculation.
                         Defaults to 1 day after last transaction.
    Returns:
        DataFrame with CustomerID, recency, frequency, monetary,
        cluster, cluster_label
    """
    if snapshot_date is None:
        snapshot_date = transactions_df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = transactions_df.groupby("CustomerID").agg(
        recency   = ("InvoiceDate",  lambda x: (snapshot_date - x.max()).days),
        frequency = ("InvoiceNo",    "nunique"),
        monetary  = ("Revenue",      "sum")
    ).reset_index()

    rfm["log_recency"]   = np.log1p(rfm["recency"])
    rfm["log_frequency"] = np.log1p(rfm["frequency"])
    rfm["log_monetary"]  = np.log1p(rfm["monetary"])

    features = ["log_recency", "log_frequency", "log_monetary"]
    X = scaler.transform(rfm[features])

    rfm["cluster"]       = kmeans.predict(X)
    rfm["cluster_label"] = rfm["cluster"].map(cluster_label_map)

    return rfm.drop(columns=["log_recency", "log_frequency", "log_monetary"])
