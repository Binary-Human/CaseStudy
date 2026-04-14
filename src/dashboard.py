"""
    This module contains functions for creating and managing the streamlit dashboard.
"""
from data import extract_context_data, extract_use_case_data


# PARAMS

file_path = "Version Candidat-ShipEasy_Dataset_CasPratique.xlsx"
sheet_name="📚 Base de Connaissance"
context_data = extract_context_data(file_path, sheet_name)

sheet_name="📋 Version Candidat"
use_case_data = extract_use_case_data(file_path, sheet_name)