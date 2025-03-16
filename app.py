import streamlit as st
import pandas as pd
from main import classify_and_add_column_to_csv, classify_all_texts

# Wrapper for main functions
def process_csv(input_csv: str, output_csv: str) -> pd.DataFrame:
    """Example function to process CSV data. Modify as needed."""
    classify_and_add_column_to_csv(input_csv, output_csv, classify_all_texts(input_csv))

st.title("CSV Processor App")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

input_csv = "input.csv"
output_csv = "output.csv"

if uploaded_file is not None:
    with open(input_csv, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    process_csv(input_csv, output_csv)
    
    
    with open(output_csv, "rb") as f:
        st.download_button(
            label="Download Processed CSV",
            data=f,
            file_name="processed_output.csv",
            mime="text/csv"
        )
