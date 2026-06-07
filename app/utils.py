import pandas as pd
import numpy as np
import joblib
import json
import duckdb
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / "src"))

from pipeline import predict_segments
from llm import generate_all_narratives

MODELS_DIR = ROOT / "src" / "models"
DATA_DIR   = ROOT / "data"


def load_models():
    kmeans      = joblib.load(MODELS_DIR / "kmeans_k4.pkl")
    scaler      = joblib.load(MODELS_DIR / "scaler.pkl")
    churn_model = joblib.load(MODELS_DIR / "churn_model.pkl")
    clv_model   = joblib.load(MODELS_DIR / "clv_model.pkl")
    next_model  = joblib.load(MODELS_DIR / "next_purchase_model.pkl")

    with open(MODELS_DIR / "cluster_label_map.json") as f:
        cluster_label_map = {int(k): v for k, v in json.load(f).items()}

    with open(MODELS_DIR / "feature_names.json") as f:
        feature_names = json.load(f)

    return {
        "kmeans":             kmeans,
        "scaler":             scaler,
        "churn_model":        churn_model,
        "clv_model":          clv_model,
        "next_model":         next_model,
        "cluster_label_map":  cluster_label_map,
        "feature_names":      feature_names,
    }


def clean_transactions(df):
    df = df.dropna(subset=["CustomerID"])
    df = df.drop_duplicates()
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]
    df["CustomerID"]  = df["CustomerID"].astype(int).astype(str)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Revenue"]     = df["Quantity"] * df["UnitPrice"]
    df = df[df["Country"] == "United Kingdom"].copy()
    return df


def build_segment_summary(rfm_df):
    return rfm_df.groupby("cluster_label").agg(
        num_customers = ("CustomerID",    "count"),
        avg_recency   = ("recency",       "mean"),
        avg_frequency = ("frequency",     "mean"),
        avg_monetary  = ("monetary",      "mean"),
    ).round(2).reset_index()


def build_ml_features(df_clean):
    snapshot = df_clean["InvoiceDate"].max() + pd.Timedelta(days=1)
    con = duckdb.connect()
    con.register("txn", df_clean)

    features = con.execute(f"""
        SELECT
            CustomerID,
            DATEDIFF('day', MAX(InvoiceDate),
                CAST('{snapshot}' AS TIMESTAMP))        AS recency,
            COUNT(DISTINCT InvoiceNo)                    AS frequency,
            ROUND(SUM(Revenue), 2)                       AS monetary,
            ROUND(AVG(Revenue), 2)                       AS avg_order_value,
            ROUND(SUM(Quantity), 0)                      AS total_items,
            COUNT(DISTINCT StockCode)                    AS unique_products,
            DATEDIFF('day', MIN(InvoiceDate),
                MAX(InvoiceDate))                        AS customer_age_days,
            ROUND(SUM(Revenue) /
                COUNT(DISTINCT InvoiceNo), 2)            AS revenue_per_order
        FROM txn
        GROUP BY CustomerID
    """).df()

    return features


def run_ml_predictions(features_df, models):
    X = features_df[models["feature_names"]]

    features_df["churn_probability"]    = models["churn_model"].predict_proba(X)[:, 1].round(3)
    features_df["clv_predicted_revenue"] = np.expm1(
        models["clv_model"].predict(X)
    ).round(2)
    features_df["next_purchase_propensity"] = models["next_model"].predict_proba(X)[:, 1].round(3)

    return features_df


def run_full_pipeline(df_raw, models):
    df_clean    = clean_transactions(df_raw)
    rfm_df      = predict_segments(df_clean)
    features_df = build_ml_features(df_clean)
    features_df = run_ml_predictions(features_df, models)

    result = features_df.merge(
        rfm_df[["CustomerID", "cluster_label"]],
        on="CustomerID", how="left"
    )
    result = result.rename(columns={"cluster_label": "segment"})

    segment_summary = build_segment_summary(rfm_df)

    return df_clean, rfm_df, result, segment_summary


def load_sample_cache():
    rfm_df     = pd.read_csv(DATA_DIR / "processed" / "rfm_clustered.csv")
    ml_preds   = pd.read_csv(DATA_DIR / "processed" / "ml_predictions.csv")

    with open(DATA_DIR / "processed" / "segment_narratives.json") as f:
        narratives = json.load(f)

    segment_summary = build_segment_summary(rfm_df)

    return rfm_df, ml_preds, narratives, segment_summary