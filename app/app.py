import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "app"))

from utils import load_models, run_full_pipeline, load_sample_cache

COLORS = {
    "Champions":       "#1B2B35",
    "Loyal Customers": "#2C5F7A",
    "At Risk":         "#B86A2E",
    "Lost":            "#8A2E2E",
}
SEGMENT_ORDER = ["Champions", "Loyal Customers", "At Risk", "Lost"]

st.set_page_config(
    page_title="Customer Segmentation",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background-color: #1B2B35 !important;
    }
    [data-testid="stSidebar"] * {
        color: #C8D8E0 !important;
    }
    [data-testid="stSidebar"] strong {
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] a {
        color: #7BB8CC !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #2E4A58 !important;
    }
    .stButton > button {
        background-color: #2C5F7A !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 7px !important;
        font-weight: 600 !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        background-color: #1B4A61 !important;
    }
    .stButton > button p {
        color: #FFFFFF !important;
    }
    /* Fix table header visibility */
    [data-testid="stDataFrame"] th {
        color: #1B2B35 !important;
        background-color: #E8E2D9 !important;
        font-weight: 600 !important;
    }
    [data-testid="stDataFrame"] td {
        color: #2A2A2A !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_models():
    return load_models()


@st.cache_data
def get_sample_data():
    return load_sample_cache()


def init_state():
    defaults = {
        "data_loaded":     False,
        "rfm_df":          None,
        "ml_preds":        None,
        "segment_summary": None,
        "narratives":      None,
        "data_source":     None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()
models = get_models()


# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.title("Customer Segmentation")
    st.divider()
    page = st.radio(
        "Go to",
        ["About", "Upload Data", "Customer Segments", "ML Predictions"],
        label_visibility="collapsed"
    )
    st.divider()
    if st.session_state["data_loaded"]:
        st.caption("ACTIVE DATASET")
        st.write(f"**Source:** {st.session_state['data_source']}")
        st.write(f"**Customers:** {len(st.session_state['rfm_df']):,}")
        st.write(f"**Segments:** 4")
    st.divider()
    st.caption("Built by [Tanmayi Shurpali](https://github.com/tanmayi123)")


# ── About ──────────────────────────────────────────────────
if page == "About":
    st.title("Customer Segmentation Platform")
    st.write(
        "This platform demonstrates a full end-to-end customer analytics pipeline built on the UCI Online Retail dataset" \
        "covering data engineering, RFM analysis, K-Means clustering, three supervised ML models, and LLM-generated segment narratives using the Gemini API"
    )
    st.divider()

    st.subheader("Dataset")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Transactions", "541,909")
    c2.metric("Customers", "3,920")
    c3.metric("Products", "3,645")
    c4.metric("Date start", "Dec 2010")
    c5.metric("Date end", "Dec 2011")
    st.write(" ")
    st.write(
        "Dataset sourced from a UK-based non-store online retailer specialising in unique all-occasion gifts. " \
        "Filtered to UK transactions only, yielding 349,203 clean records after removing cancellations, missing customer IDs, and zero-value entries"
    )
    st.divider()

    st.subheader("Pipeline")
    with st.expander("1. Data cleaning and feature engineering"):
        st.write(
            "Cancellations, missing CustomerIDs, zero quantities, and duplicates are removed. "
            "Revenue is computed as Quantity x UnitPrice. Eight customer-level features are "
            "built using DuckDB SQL queries on pandas dataframes: recency, frequency, monetary "
            "value, average order value, total items, unique products, customer age in days, "
            "and revenue per order."
        )
    with st.expander("2. RFM segmentation"):
        st.write(
            "RFM features are log-transformed to correct for right skew, then standardised "
            "with StandardScaler. K-Means clustering is applied with k=4, selected via elbow "
            "method and silhouette scoring across k=2 to k=10. Silhouette score at k=4 is 0.34 "
            "with clearly separable business segments."
        )
    with st.expander("3. ML models"):
        st.write(
            "Three supervised models trained on a 9-month observation window (Dec 2010 to "
            "Sep 2011) with labels from the final 3 months (Oct 2011 to Dec 2011)."
        )
        st.write(" ")
        col1, col2, col3 = st.columns(3, gap="medium")
        with col1:
            st.markdown("**Churn Prediction**")
            st.caption("Random Forest Classifier")
            st.write("Label: No purchase in final 3 months")
            st.write("Class balance: 49% / 51%")
            st.write("ROC-AUC: 0.727")
            st.write("Top features: Monetary, frequency, customer age")
        with col2:
            st.markdown("**CLV Scoring**")
            st.caption("Gradient Boosting Regressor")
            st.write("Label: Revenue in final 3 months")
            st.write("Target: log1p transformed")
            st.write("MAE: £362.78")
            st.write("Top features: Frequency, monetary, unique products")
        with col3:
            st.markdown("**Next Purchase Propensity**")
            st.caption("Gradient Boosting Classifier")
            st.write("Label: Purchase within 30 days")
            st.write("Class balance: 27% / 73%")
            st.write("ROC-AUC: 0.713")
            st.write("Top features: Frequency, recency, monetary")
    with st.expander("4. LLM integration"):
        st.write(
            "Segment narratives generated using Gemini 2.5 Flash. Each segment receives a "
            "structured prompt with its RFM metrics and business context. Returns a 4-section "
            "executive report covering segment profile, revenue impact, churn risk, and "
            "recommended actions with specific channels and expected outcomes. "
            "Pre-cached for sample data, generated fresh for uploads."
        )
    st.divider()

    st.subheader("Tech stack")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("**Data and modeling**")
        st.write("Python 3.13, pandas, numpy")
        st.write("DuckDB (SQL on dataframes)")
        st.write("scikit-learn (K-Means, Random Forest, Gradient Boosting)")
        st.write("Plotly")
    with col2:
        st.markdown("**Infrastructure**")
        st.write("Google Gemini 2.5 Flash")
        st.write("Streamlit")
        st.write("Docker")
        st.write("Hugging Face Spaces")
    st.divider()
    st.write("**Tanmayi Shurpali** — [github.com/tanmayi123](https://github.com/tanmayi123)")


# ── Upload Data ────────────────────────────────────────────
elif page == "Upload Data":
    st.title("Upload Data")
    st.write(
        "Upload a transactions CSV or load the sample dataset to run the full pipeline."
    )
    st.divider()

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("Load sample Dataset")
    st.write(
        "Loads the UCI Online Retail dataset. UK transactions from Dec 2010 to Dec 2011, "
        "3,920 customers"
        "Link : [UCI Online Retail](https://archive.ics.uci.edu/ml/datasets/online+retail)"
    )
    if st.button("Load sample dataset"):
        with st.spinner("Loading..."):
            rfm_df, ml_preds, narratives, segment_summary = get_sample_data()
            st.session_state["data_loaded"]     = True
            st.session_state["rfm_df"]          = rfm_df
            st.session_state["ml_preds"]        = ml_preds
            st.session_state["segment_summary"] = segment_summary
            st.session_state["narratives"]      = narratives
            st.session_state["data_source"]     = "UCI Online Retail (sample)"
        st.success("Sample data loaded. Navigate to Customer Segments or ML Predictions.")

    if st.session_state["data_loaded"]:
        st.divider()
        st.subheader("Dataset overview")
        rfm_df          = st.session_state["rfm_df"]
        segment_summary = st.session_state["segment_summary"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total customers", f"{len(rfm_df):,}")
        c2.metric("Segments", "4")
        c3.metric("Avg monetary", f"£{rfm_df['monetary'].mean():,.0f}")
        c4.metric("Avg recency", f"{rfm_df['recency'].mean():.0f} days")

        st.write(" ")
        st.subheader("Segment breakdown")
        st.dataframe(
            segment_summary.rename(columns={
                "cluster_label": "Segment",
                "num_customers": "Customers",
                "avg_recency":   "Avg Recency (days)",
                "avg_frequency": "Avg Frequency",
                "avg_monetary":  "Avg Monetary (£)"
            }).sort_values("Customers", ascending=False),
            use_container_width=True,
            hide_index=True
        )


# ── Customer Segments ──────────────────────────────────────
elif page == "Customer Segments":
    st.title("Customer Segments")
    st.write(
        "Segmentation is driven by RFM analysis, where recency, frequency, and monetary features are log-transformed and standardised before being fed into a K-Means clustering model (k=4) to identify four meaningful customer groups: Champions, Loyal Customers, At Risk, and Lost. Each segment's RFM profile is visualised and accompanied by an AI-generated narrative providing business insights and recommended actions."
    )
    st.divider()

    if not st.session_state["data_loaded"]:
        st.info("No data loaded. Go to Upload Data to get started.")
    else:
        segment_summary = st.session_state["segment_summary"]
        rfm_df          = st.session_state["rfm_df"]
        narratives      = st.session_state["narratives"]

        ordered = segment_summary.set_index("cluster_label").reindex(
            [s for s in SEGMENT_ORDER if s in segment_summary["cluster_label"].values]
        ).reset_index()

        st.subheader("Segment distribution")
        col1, col2 = st.columns(2, gap="large")

        with col1:
            fig = px.bar(
                ordered,
                x="cluster_label", y="num_customers",
                color="cluster_label", color_discrete_map=COLORS,
                labels={"cluster_label": "Segment", "num_customers": "Customers"},
                title="Customers per segment"
            )
            fig.update_layout(
                showlegend=False,
                plot_bgcolor="#F0EBE3",
                paper_bgcolor="#F0EBE3",
                font_color="#1A1A1A",
                title_font_size=14
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.pie(
                ordered,
                names="cluster_label", values="num_customers",
                color="cluster_label", color_discrete_map=COLORS,
                title="Segment share"
            )
            fig2.update_layout(
                plot_bgcolor="#F0EBE3",
                paper_bgcolor="#F0EBE3",
                font_color="#1A1A1A",
                title_font_size=14
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("RFM metrics by segment")
        col1, col2, col3 = st.columns(3, gap="medium")

        for col, metric, label in zip(
            [col1, col2, col3],
            ["avg_recency", "avg_frequency", "avg_monetary"],
            ["Avg Recency (days)", "Avg Frequency", "Avg Monetary (£)"]
        ):
            fig = px.bar(
                ordered,
                x="cluster_label", y=metric,
                color="cluster_label", color_discrete_map=COLORS,
                labels={"cluster_label": "Segment", metric: label},
                title=label
            )
            fig.update_layout(
                showlegend=False,
                plot_bgcolor="#F0EBE3",
                paper_bgcolor="#F0EBE3",
                font_color="#1A1A1A",
                title_font_size=14
            )
            col.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Segment metrics table")
        st.dataframe(
            ordered.rename(columns={
                "cluster_label": "Segment",
                "num_customers": "Customers",
                "avg_recency":   "Avg Recency (days)",
                "avg_frequency": "Avg Frequency",
                "avg_monetary":  "Avg Monetary (£)"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.divider()
        st.subheader("Segment analysis")
        st.write("The analysis below is LLM-powered, with Gemini producing business insights for each customer segment based on their RFM profile")

        selected = st.selectbox(
            "Select segment",
            [s for s in SEGMENT_ORDER if s in narratives]
        )

        if selected and narratives:
            row = segment_summary[
                segment_summary["cluster_label"] == selected
            ].iloc[0]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Customers",     f"{row['num_customers']:,}")
            c2.metric("Avg Recency",   f"{row['avg_recency']} days")
            c3.metric("Avg Frequency", f"{row['avg_frequency']} orders")
            c4.metric("Avg Monetary",  f"£{row['avg_monetary']:,.0f}")

            st.write(" ")
            with st.container(border=True):
                st.markdown(narratives[selected])


# ── ML Predictions ─────────────────────────────────────────
elif page == "ML Predictions":
    st.title("ML Predictions")
    st.write(
        "Three machine learning models were trained on a 9-month observation window and validated against the final 3 months of data." \
        "Each model uses features derived from RFM metrics and transaction history to predict future customer behavior. The Churn Prediction model identifies customers at risk of leaving, the CLV Scoring model estimates future revenue potential, and the Next Purchase Propensity model assesses likelihood of a purchase within 30 days. Pre-computed predictions are available for the sample dataset, while uploads will trigger on-the-fly inference with the loaded models."
    )
    st.divider()

    if not st.session_state["data_loaded"]:
        st.info("No data loaded. Go to Upload Data to get started.")
    else:
        ml_preds = st.session_state["ml_preds"].copy()

        st.subheader("Model performance")
        c1, c2, c3 = st.columns(3)
        c1.metric("Churn Model ROC-AUC",    "0.727")
        c2.metric("CLV Model MAE",          "£362")
        c3.metric("Next Purchase ROC-AUC",  "0.713")

        st.divider()
        st.subheader("Customer predictions")

        col1, col2 = st.columns([1, 3], gap="medium")
        with col1:
            segments_available = ["All"] + [
                s for s in SEGMENT_ORDER
                if s in ml_preds.get("segment", pd.Series()).unique()
            ]
            selected_segment = st.selectbox("Filter by segment", segments_available)
            churn_threshold = st.slider(
                "Min churn probability",
                min_value=0.0, max_value=1.0, value=0.0, step=0.05
            )

        filtered = ml_preds.copy()
        if selected_segment != "All":
            filtered = filtered[filtered["segment"] == selected_segment]
        if "churn_probability" in filtered.columns:
            filtered = filtered[filtered["churn_probability"] >= churn_threshold]

        with col2:
            st.write(f"Showing **{len(filtered):,}** customers")

        display_cols = [
            "CustomerID", "segment", "recency", "frequency", "monetary",
            "churn_probability", "clv_predicted_revenue", "next_purchase_propensity"
        ]
        available_cols = [c for c in display_cols if c in filtered.columns]

        st.dataframe(
            filtered[available_cols].rename(columns={
                "CustomerID":               "Customer ID",
                "segment":                  "Segment",
                "recency":                  "Recency (days)",
                "frequency":                "Frequency",
                "monetary":                 "Monetary (£)",
                "churn_probability":        "Churn Probability",
                "clv_predicted_revenue":    "Predicted CLV (£)",
                "next_purchase_propensity": "Next Purchase Score",
            }),
            use_container_width=True,
            hide_index=True
        )

        st.divider()
        st.subheader("Score distributions")
        col1, col2, col3 = st.columns(3, gap="medium")

        if "churn_probability" in ml_preds.columns:
            with col1:
                fig = px.histogram(
                    ml_preds, x="churn_probability", nbins=30,
                    color_discrete_sequence=["#8A2E2E"],
                    title="Churn probability",
                    labels={"churn_probability": "Score"}
                )
                fig.update_layout(
                    plot_bgcolor="#F0EBE3", paper_bgcolor="#F0EBE3",
                    showlegend=False, title_font_size=14
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig2 = px.histogram(
                    ml_preds, x="clv_predicted_revenue", nbins=30,
                    color_discrete_sequence=["#2C5F7A"],
                    title="Predicted CLV",
                    labels={"clv_predicted_revenue": "Revenue (£)"}
                )
                fig2.update_layout(
                    plot_bgcolor="#F0EBE3", paper_bgcolor="#F0EBE3",
                    showlegend=False, title_font_size=14
                )
                st.plotly_chart(fig2, use_container_width=True)

            with col3:
                fig3 = px.histogram(
                    ml_preds, x="next_purchase_propensity", nbins=30,
                    color_discrete_sequence=["#1B2B35"],
                    title="Next purchase score",
                    labels={"next_purchase_propensity": "Score"}
                )
                fig3.update_layout(
                    plot_bgcolor="#F0EBE3", paper_bgcolor="#F0EBE3",
                    showlegend=False, title_font_size=14
                )
                st.plotly_chart(fig3, use_container_width=True)

        st.divider()
        st.subheader("Export")
        csv = filtered[available_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download predictions as CSV",
            data=csv,
            file_name="customer_predictions.csv",
            mime="text/csv"
        )