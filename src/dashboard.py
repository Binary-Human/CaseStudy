"""
    Streamlit dashboard.
"""

import streamlit as st
import pandas as pd
from data import extract_context_data, extract_use_case_data

st.set_page_config(page_title="ShipEasy Dashboard", layout="wide")

st.title("📊 ShipEasy Data Explorer")

# Default sheet names
DEFAULT_CONTEXT_SHEET = "📚 Base de Connaissance"
DEFAULT_USE_CASE_SHEET = "📋 Version Candidat"

# File uploader
uploaded_file = st.file_uploader(
    "Upload Excel file",
    type=["xlsx"]
)
if uploaded_file is None:
    st.info("Please upload an Excel file to continue.")
    st.stop()

# Read available sheets
xls = pd.ExcelFile(uploaded_file)
sheet_names = xls.sheet_names

# Select which sheet is to be used
st.sidebar.header("⚙️ Parameters")

context_sheet = st.sidebar.selectbox(
    "Context Sheet",
    options=sheet_names,
    index=sheet_names.index(DEFAULT_CONTEXT_SHEET) if DEFAULT_CONTEXT_SHEET in sheet_names else 0 
    # default sheet if exists, else first sheet
)

use_case_sheet = st.sidebar.selectbox(
    "Use Case Sheet",
    options=sheet_names,
    index=sheet_names.index(DEFAULT_USE_CASE_SHEET) if DEFAULT_USE_CASE_SHEET in sheet_names else 0 
    # default sheet if exists
)

# Load data
try:
    context_data = extract_context_data(uploaded_file, context_sheet)
    use_case_data = extract_use_case_data(uploaded_file, use_case_sheet)

    # Display tables
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📚 Context Data")
        st.dataframe(context_data, use_container_width=True)

    with col2:
        st.subheader("📋 Use Case Data")
        st.dataframe(use_case_data, use_container_width=True)

except Exception as e:
    st.error(f"Error processing file: {e}")