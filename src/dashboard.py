"""
    Streamlit dashboard.
"""

import streamlit as st
import pandas as pd

from data import extract_context_data, extract_use_case_data

from evaluation.service import EvaluationService 
from evaluation.metrics import categorize


st.set_page_config(page_title="ShipEasy Dashboard", layout="wide")
st.title("📊 ShipEasy Evaluation Dashboard")

# Default sheet names
DEFAULT_CONTEXT_SHEET = "📚 Base de Connaissance"
DEFAULT_USE_CASE_SHEET = "📋 Version Candidat"

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])
if uploaded_file is None:
    st.info("Upload file to start.")
    st.stop()

xls = pd.ExcelFile(uploaded_file)
sheet_names = xls.sheet_names

##################################################
st.sidebar.header("⚙️ Parameters")

context_sheet = st.sidebar.selectbox("Context Sheet", options=sheet_names,
    index=sheet_names.index(DEFAULT_CONTEXT_SHEET) if DEFAULT_CONTEXT_SHEET in sheet_names else 0)
use_case_sheet = st.sidebar.selectbox("Use Case Sheet", options=sheet_names,
    index=sheet_names.index(DEFAULT_USE_CASE_SHEET) if DEFAULT_USE_CASE_SHEET in sheet_names else 0)

# Thresholds specification
st.sidebar.subheader("📏 Thresholds")

# TODO : update and integrate in metrics categorize
thresholds = {
    "faithfulness": st.sidebar.slider("Faithfulness", 0.0, 1.0, 0.7),
    "relevancy": st.sidebar.slider("Relevancy", 0.0, 1.0, 0.7),
    "tone": st.sidebar.slider("Tone", 0.0, 1.0, 0.8),
}

##################################################

# Load data
context_df = extract_context_data(uploaded_file, context_sheet)
use_case_df = extract_use_case_data(uploaded_file, use_case_sheet)

# Display tables
col1, col2 = st.columns(2)

with col1:
    st.subheader("📚 Context Data")
    st.dataframe(context_df, use_container_width=True)

with col2:
    st.subheader("📋 Use Case Data")
    st.dataframe(use_case_df, use_container_width=True)

##################################################

# Run evaluation
if st.button("🚀 Run Evaluation"):

    service = EvaluationService(context_df)
    results = []
    progress = st.progress(0)

    for i, row in use_case_df.iterrows():

        print(f"Evaluating case {i+1}/{len(use_case_df)}: {row['question']}")
        metrics = service.evaluate_case(row)
        label = categorize(metrics, thresholds)

        # Metrics
        results.append({
            "id": row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "label": label,
            "relevancy": metrics.get("relevancy", 0),
            "faithfulness": metrics.get("faithfulness", 0),
            "hallucination": metrics.get("hallucination", 0),
            "refusal": metrics.get("refusal", 0),
            "tone": metrics.get("tone", 0),
        })

        progress.progress((i + 1) / len(use_case_df))

    df_results = pd.DataFrame(results)

    st.success("Evaluation complete")
    st.subheader("📊 Summary")

    col1, col2, col3, col4, col5 = st.columns(5)

    # Detailed count
    col1.metric("Total cases", len(df_results))
    col2.metric("Hallucinations", (df_results["label"] == "🔴 Hallucination").sum())
    col3.metric("Refusals", (df_results["label"] == "🔵 Refus inapproprié").sum())
    col4.metric("Tone issues", (df_results["label"] == "🟣 Bon fond, mauvais ton").sum())
    col5.metric("Good responses", (df_results["label"] == "✅ Bonne réponse").sum())

    st.subheader("📈 Label distribution")
    st.bar_chart(df_results["label"].value_counts())

    st.subheader("❌ Failed cases")
    failed = df_results[df_results["label"] != "✅ Bonne réponse"]
    st.dataframe(failed, use_container_width=True)

    st.subheader("🚨 Most critical issues")
    critical = failed.sort_values(
        by=["relevancy"], # Ascending for relevancy, descending for others
        ascending=True
    ).head(10)

    st.subheader("🚨 Most critical issues")
    critical = failed.sort_values(
        by=["hallucination", "refusal", "relevancy"], # Ascending for relevancy, descending for others
        ascending=[False, False, True]
    ).head(10)

    st.dataframe(critical, use_container_width=True)

    st.download_button(
        "Download results",
        df_results.to_csv(index=False),
        file_name="eval_results.csv"
    )
