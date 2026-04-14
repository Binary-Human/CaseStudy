"""
    Functions for loading and processing data for the application.
"""
import pandas as pd

# Extract context data from .xlsx file
def extract_context_data(file_path, sheet_name):

    # Load sheet
    sheet_raw = pd.read_excel(file_path, sheet_name)

    # Row 1 is a header - fix
    data = sheet_raw.iloc[1:].copy()
    data.columns = ["category", "rule", "detail", "source"]

    # Reset index
    data = data.reset_index(drop=True)

    print(data.head())

    return data


# Extract use case data from .xlsx file
def extract_use_case_data(file_path, sheet_name):
    # Load dataset
    sheet_raw = pd.read_excel(file_path, sheet_name)

    # Skip intro rows
    data = sheet_raw.iloc[2:].copy()
    data.columns = ["id", "question", "answer", "eval", "notes"]

    # Reset index
    data = data.reset_index(drop=True)

    print(data.head())

    return data
