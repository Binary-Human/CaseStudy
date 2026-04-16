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

thresholds = {
    "faithfulness": st.sidebar.slider("Faithfulness", 0.0, 1.0, 0.9),
    "relevancy": st.sidebar.slider("Relevancy", 0.0, 1.0, 0.9),
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

    # To avoid rerun 
    st.session_state["df_results"] = pd.DataFrame(results)


if "df_results" in st.session_state:
    df_results = st.session_state["df_results"]

    st.success("Evaluation complete")
    st.header("📊 Summary")

    col1, col2, col3, col4, col5 = st.columns(5)

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

    st.subheader("🔍 Detailed metrics distribution")
    selected_label = st.selectbox(
        "Select label to filter",
        options=df_results["label"].unique()
    )

    filtered = df_results[df_results["label"] == selected_label]

    st.dataframe(
        filtered[[
            "question", "answer",
            "relevancy", 
            "faithfulness",
            "hallucination",
            "refusal", 
            "tone"
        ]],
        use_container_width=True
    )

    st.download_button(
        "Download results",
        df_results.to_csv(index=False),
        file_name="eval_results.csv"
    )

##################################################
st.header("🆚 Compare Runs")

"""Upload your csv runs"""

col_left, col_right = st.columns(2)

with col_left:
    file_run_1 = st.file_uploader("Upload Run 1 CSV", type=["csv"], key="run1")

with col_right:
    file_run_2 = st.file_uploader("Upload Run 2 CSV", type=["csv"], key="run2")

if file_run_1 and file_run_2:

    df1 = pd.read_csv(file_run_1)
    df2 = pd.read_csv(file_run_2)

    # Align on ID
    merged = df1.merge(df2, on="id", suffixes=("_run1", "_run2"))

    st.success("Comparison ready")

    ##################################################
    st.subheader("📊 Global Statistics")

    col1, col2, col3, col4 = st.columns(4)

    # How many good responses in each run
    col1.metric("Run1 Good", (df1["label"] == "✅ Bonne réponse").sum())
    col2.metric("Run2 Good", (df2["label"] == "✅ Bonne réponse").sum())

    # How many got worse
    improvement = (
        (merged["label_run1"] != "✅ Bonne réponse") &
        (merged["label_run2"] == "✅ Bonne réponse")
    ).sum()

    # How many got better
    regression = (
        (merged["label_run1"] == "✅ Bonne réponse") &
        (merged["label_run2"] != "✅ Bonne réponse")
    ).sum()

    col3.metric("⬆️ Improvements", improvement)
    col4.metric("⬇️ Regressions", regression)

    ##################################################
    st.subheader("📈 Label Distribution Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Run 1")
        st.bar_chart(df1["label"].value_counts())

    with col2:
        st.write("Run 2")
        st.bar_chart(df2["label"].value_counts())

    #################]#################################
    st.subheader("🔍 Detailed Changes")

    changed = merged[merged["label_run1"] != merged["label_run2"]]

    st.dataframe(changed[[
        "id",
        "question_run1",
        "answer_run1",
        "answer_run2",
        "label_run1",
        "label_run2"
    ]], use_container_width=True)