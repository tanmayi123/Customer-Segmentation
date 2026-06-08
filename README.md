# Customer Segmentation Platform

---
title: Customer Segmentation
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

An end-to-end customer analytics platform that takes raw e-commerce transaction data and produces RFM-based customer segments, three supervised ML model predictions, and LLM-generated business narratives per segment. Built on the UCI Online Retail dataset and deployable as a Streamlit web app via Docker and Hugging Face Spaces.

---

## Live Demo

[Link to be added after Hugging Face deployment]

---

## Project Structure

```
Customer-Segmentation/
├── data/
│   ├── raw/                              # Raw input data (UCI Online Retail.xlsx)
│   └── processed/                        # Cleaned data, RFM scores, ML predictions, narratives
├── notebooks/
│   ├── 01_eda_and_rfm.ipynb              # EDA, DuckDB SQL analysis, RFM segmentation
│   ├── 02_prediction_pipeline.ipynb      # K-Means model training and prediction pipeline
│   ├── 03_ml_models.ipynb                # Churn, CLV, and next-purchase models
│   └── 04_llm_integration.ipynb         # Gemini API segment narrative generation
├── src/
│   ├── pipeline.py                       # Reusable RFM and segmentation pipeline module
│   ├── llm.py                            # Gemini API narrative generation module
│   └── models/                           # Saved model artifacts (.pkl, .json)
├── app/
│   ├── app.py                            # Streamlit application
│   └── utils.py                          # Pipeline orchestration and utility functions
├── .streamlit/
│   └── config.toml                       # Streamlit theme configuration
├── Dockerfile                            # Container definition for deployment
├── requirements.txt                      # Python dependencies
└── README.md
```

---

## What It Does

### 1. Data cleaning and feature engineering

Raw transactions are cleaned by removing cancellations (invoices prefixed with C), rows with missing CustomerID, zero or negative quantities, and duplicate records. Revenue is computed as Quantity x UnitPrice. Eight customer-level features are engineered using DuckDB SQL queries running directly on pandas dataframes:

- Recency: days since last purchase
- Frequency: number of distinct orders
- Monetary: total lifetime spend
- Average order value
- Total items purchased
- Unique products purchased
- Customer age in days
- Revenue per order

### 2. RFM segmentation

RFM features are log-transformed to correct for right skew, then standardised using StandardScaler. K-Means clustering is applied with k=4, selected based on elbow method and silhouette scoring across k=2 to k=10. The optimal k=4 yields a silhouette score of 0.34 with four clearly interpretable business segments: Champions, Loyal Customers, At Risk, and Lost.

### 3. ML models

Three supervised models are trained on a 9-month observation window (Dec 2010 to Sep 2011) with labels derived from the final 3 months (Oct 2011 to Dec 2011).

| Model | Type | Label | Performance |
|---|---|---|---|
| Churn prediction | Random Forest Classifier | No purchase in final 3 months | ROC-AUC: 0.727 |
| CLV scoring | Gradient Boosting Regressor | Revenue in final 3 months | MAE: £362.78 |
| Next purchase propensity | Gradient Boosting Classifier | Purchase within next 30 days | ROC-AUC: 0.713 |

### 4. LLM integration

Segment narratives are generated using the Gemini 2.5 Flash API. Each segment receives a structured prompt containing its RFM metrics, segment size as a percentage of the total customer base, and business context. The model returns a 4-section executive report covering segment profile, revenue impact, churn and retention risk, and recommended actions with specific channels and expected outcomes tied to RFM metrics.

---

## Dataset

**UCI Online Retail Dataset**

- 541,909 transactions
- 3,920 unique customers (UK only after filtering)
- 3,645 unique products
- Date range: December 2010 to December 2011
- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online+retail)

The dataset covers a UK-based non-store online retailer specialising in unique all-occasion gifts. Analysis is filtered to UK transactions only, yielding 349,203 clean records.

**Required columns for upload:**

```
InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country
```

**Compatible test dataset:**

[Online Retail II UCI](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci) on Kaggle. Same retailer, two-year version (2009-2011), identical column structure.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data processing | Python 3.13, pandas, numpy |
| SQL analysis | DuckDB |
| ML modeling | scikit-learn (K-Means, Random Forest, Gradient Boosting) |
| Visualisation | Plotly |
| LLM | Google Gemini 2.5 Flash |
| App | Streamlit |
| Containerisation | Docker |
| Deployment | Hugging Face Spaces |
| Version control | GitHub |

---

## Setup and Installation

### Prerequisites

- Python 3.10 or higher
- A Google Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

### Local setup

```bash
# Clone the repository
git clone https://github.com/tanmayi123/Customer-Segmentation.git
cd Customer-Segmentation

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Add your Gemini API key
echo "GEMINI_API_KEY=your_key_here" > .env
```

### Download the dataset

Download the UCI Online Retail dataset from [here](https://archive.ics.uci.edu/dataset/352/online+retail) and place the Excel file at:

```
data/raw/Online Retail.xlsx
```

### Run the notebooks

Run the notebooks in order to reproduce the full pipeline:

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
# Build the image
docker build -t customer-segmentation .

# Run the container
docker run -p 8501:8501 -e GEMINI_API_KEY=your_key_here customer-segmentation
```

---

## Notebooks Overview

### 01_eda_and_rfm.ipynb

- Data loading and cleaning
- Exploratory data analysis using DuckDB SQL queries on pandas dataframes
- Monthly revenue trends, top products, revenue by day of week
- RFM feature computation and scoring
- Segment labelling and visualisation

### 02_prediction_pipeline.ipynb

- Log transformation and StandardScaler feature prep
- Elbow method and silhouette scoring for optimal k selection
- K-Means model training with k=4
- Cluster labelling and visualisation
- Reusable predict_segments() pipeline function saved to src/pipeline.py

### 03_ml_models.ipynb

- Observation and prediction window construction
- Feature engineering via DuckDB
- Churn label: binary (purchased in final 3 months or not)
- CLV label: continuous (total revenue in final 3 months)
- Next purchase label: binary (purchased within 30 days of observation end)
- Model training, evaluation, and ROC curve analysis
- Feature importance analysis for all three models

### 04_llm_integration.ipynb

- Gemini 2.5 Flash API integration
- Structured prompt engineering for executive-level segment reports
- Batch narrative generation for all four segments
- Narratives saved to data/processed/segment_narratives.json

---

## Segment Definitions

| Segment | Description |
|---|---|
| Champions | High recency, high frequency, highest spend. Core revenue drivers. |
| Loyal Customers | Moderate recency, moderate-high frequency, solid spend. Retention focus. |
| At Risk | Recent but low frequency and low spend. New customers not yet converted. |
| Lost | High recency (long time since purchase), low frequency, low spend. Re-engagement needed. |

---

## Author

**Tanmayi Shurpali**

[github.com/tanmayi123](https://github.com/tanmayi123)