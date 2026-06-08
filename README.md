---
title: Customer Segmentation
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Customer Segmentation Platform

An end-to-end customer analytics platform built on the UCI Online Retail dataset. The platform takes raw e-commerce transaction data and produces RFM-based customer segments, three supervised ML model predictions, and LLM-generated business narratives per segment. Deployed as a Streamlit web app via Docker on Hugging Face Spaces.

**Live demo:** [huggingface.co/spaces/TanmayiShurpali/customer-segmentation](https://huggingface.co/spaces/TanmayiShurpali/customer-segmentation)

---

## Overview

Most customer analytics pipelines stop at segmentation. This platform goes further by combining unsupervised clustering with three supervised ML models and an LLM layer that translates segment statistics into actionable business narratives. The result is a self-contained analytics tool that takes a raw transactions file and produces strategic recommendations without any manual interpretation.

The full pipeline runs in under a minute on a new dataset and supports both a pre-loaded sample mode and a live upload mode where users supply their own transaction data.

---

## Live Demo

The app is deployed on Hugging Face Spaces and publicly accessible:

[https://huggingface.co/spaces/TanmayiShurpali/customer-segmentation](https://huggingface.co/spaces/TanmayiShurpali/customer-segmentation)

To explore, use the "Load sample dataset" button on the Upload Data page. This loads the pipeline and shows results for the UCI Online Retail dataset.
---

## What It Does

### 1. Data Cleaning and Feature Engineering

Raw transaction files are cleaned by removing cancellations (invoices prefixed with C), records with missing CustomerID, zero or negative quantities, and exact duplicate rows. Revenue is computed as Quantity multiplied by UnitPrice. The pipeline then engineers eight customer-level features using SQL queries running directly on in-memory dataframes:

- Recency: days since the customer's last purchase relative to a snapshot date
- Frequency: count of distinct invoice numbers
- Monetary: total lifetime revenue
- Average order value: mean revenue per invoice
- Total items: sum of quantities purchased
- Unique products: count of distinct stock codes purchased
- Customer age in days: span between first and last purchase date
- Revenue per order: total revenue divided by order count

### 2. RFM Segmentation

RFM features are log-transformed to correct for right skew and then standardised using StandardScaler. K-Means clustering is applied with k=4, selected based on the elbow method and silhouette scoring evaluated across k=2 to k=10. The optimal k=4 yields a silhouette score of 0.34 with four clearly separable business segments.

Cluster labels are assigned by ranking clusters on average monetary value, producing the following segments:

| Segment | Customers | Avg Recency | Avg Frequency | Avg Monetary |
|---|---|---|---|---|
| Champions | 636 | 11.86 days | 13.6 orders | £7,192 |
| Loyal Customers | 1,050 | 64.86 days | 4.29 orders | £1,749 |
| At Risk | 823 | 22.94 days | 1.94 orders | £476 |
| Lost | 1,411 | 190.66 days | 1.35 orders | £342 |

### 3. ML Models

Three supervised models are trained on a 9-month observation window (December 2010 to September 2011) with labels derived from the final 3 months (October 2011 to December 2011). All three models use the same 8 engineered features.

**Churn Prediction (Random Forest Classifier)**
Predicts whether a customer will not purchase during the prediction window. The dataset has 48.6% positive churn rate. ROC-AUC of 0.727. Top features by importance are monetary value, frequency, customer age, and recency.

**Customer Lifetime Value Scoring (Gradient Boosting Regressor)**
Predicts total revenue a customer will generate in the prediction window. The target is log1p-transformed to reduce skew. MAE of £362.78 and R-squared of 0.23 on holdout. Top features are frequency, monetary value, and unique products.

**Next Purchase Propensity (Gradient Boosting Classifier)**
Predicts whether a customer will purchase within 30 days of the observation window end. 26.5% positive rate. ROC-AUC of 0.713. Top features are frequency, recency, monetary value, and total items.

### 4. LLM Integration

Segment narratives are generated using the Gemini 2.5 Flash API. Each segment receives a structured prompt containing its name, size as a percentage of the total customer base, and average RFM values, along with business context about the retailer type and dataset. The model returns a structured 4-section executive report covering:

- Segment profile: behavioural interpretation of the RFM metrics
- Revenue impact: estimated total and relative contribution
- Churn and retention risk: assessment based on recency relative to expected purchase cycles
- Recommended actions: three specific tactics each with a channel, targeting rationale, and expected impact on RFM metrics

For the sample dataset, narratives are pre-cached and load instantly. For uploaded data, narratives are generated fresh via the API.

---

## App Pages

**About**
Project overview, dataset statistics, pipeline documentation, and tech stack.

**Upload Data**
File uploader accepting transaction CSV files, plus a one-click button to load the pre-computed sample dataset. Displays dataset summary on load.

**Customer Segments**
Segment distribution bar and pie charts, RFM metric comparisons across segments, a summary metrics table, and the Gemini-generated narrative for each segment selectable via dropdown.

**ML Predictions**
Model performance metrics, a filterable customer table with all three prediction scores, score distribution histograms, and a CSV export of the filtered results.

---

## Dataset

**UCI Online Retail Dataset**

- 541,909 raw transactions
- Cleaned to 349,203 UK transactions
- 3,920 unique customers
- 3,645 unique products
- Date range: December 2010 to December 2011
- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online+retail)

The dataset covers a UK-based non-store online retailer specialising in unique all-occasion gifts. Analysis is filtered to UK transactions only after cleaning.

**Required columns for uploaded files:**

```
InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country
```

---

## Project Structure

```
Customer-Segmentation/
├── data/
│   ├── raw/
│   │   └── Online Retail.xlsx
│   └── processed/
│       ├── transactions_clean.csv
│       ├── rfm_scored.csv
│       ├── rfm_clustered.csv
│       ├── ml_predictions.csv
│       └── segment_narratives.json
├── notebooks/
│   ├── 01_eda_and_rfm.ipynb
│   ├── 02_prediction_pipeline.ipynb
│   ├── 03_ml_models.ipynb
│   └── 04_llm_integration.ipynb
├── src/
│   ├── pipeline.py
│   ├── llm.py
│   └── models/
│       ├── kmeans_k4.pkl
│       ├── scaler.pkl
│       ├── churn_model.pkl
│       ├── clv_model.pkl
│       ├── next_purchase_model.pkl
│       ├── cluster_label_map.json
│       └── feature_names.json
├── app/
│   ├── app.py
│   └── utils.py
├── .streamlit/
│   └── config.toml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Notebooks

**01_eda_and_rfm.ipynb**
Data loading, cleaning, and exploratory analysis. SQL-based analysis covering monthly revenue trends, top products by revenue, and revenue by day of week. RFM feature computation using NTILE scoring, segment labelling, and distribution visualisation.

**02_prediction_pipeline.ipynb**
Log transformation and StandardScaler feature prep. Elbow method and silhouette scoring for k selection across k=2 to k=10. K-Means training with k=4, cluster labelling by monetary rank, and reusable predict_segments() function saved to src/pipeline.py.

**03_ml_models.ipynb**
Observation and prediction window construction. Feature engineering for 3,253 customers in the observation window. Training and evaluation of all three models with ROC curve analysis and feature importance plots.

**04_llm_integration.ipynb**
Gemini 2.5 Flash API integration. Structured prompt engineering for executive-level segment reports. Batch narrative generation for all four segments and export to segment_narratives.json.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data processing | Python 3.13, pandas, numpy |
| SQL analysis | In-memory SQL on dataframes |
| ML modeling | scikit-learn (K-Means, Random Forest, Gradient Boosting) |
| Visualisation | Plotly |
| LLM | Google Gemini 2.5 Flash |
| App framework | Streamlit |
| Containerisation | Docker |
| Deployment | Hugging Face Spaces |
| Version control | GitHub |

---

## Local Setup

### Prerequisites

- Python 3.10 or higher
- A Google Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

### Installation

```bash
git clone https://github.com/tanmayi123/Customer-Segmentation.git
cd Customer-Segmentation

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

echo "GEMINI_API_KEY=your_key_here" > .env
```

### Dataset

Download the UCI Online Retail dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online+retail) and place the file at:

```
data/raw/Online Retail.xlsx
```

### Run the notebooks

Run in order to reproduce the full pipeline:

```
notebooks/01_eda_and_rfm.ipynb
notebooks/02_prediction_pipeline.ipynb
notebooks/03_ml_models.ipynb
notebooks/04_llm_integration.ipynb
```

### Run the app

```bash
streamlit run app/app.py
```

---

## Docker

```bash
docker build -t customer-segmentation .

docker run -p 7860:7860 -e GEMINI_API_KEY=your_key_here customer-segmentation
```

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
openpyxl
duckdb
scikit-learn
jupyter
ipykernel
google-genai
python-dotenv
streamlit
plotly
joblib
```