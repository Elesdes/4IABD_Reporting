import pandas as pd
import streamlit as st
from app.df_utils import generate_df_from_metadata
from app.reporting import clean_dataset

# Set page configuration
st.set_page_config(page_title="Report Generator", page_icon="chart_with_upwards_trend")

# Add title and description
st.title("Report Generator")
st.write(
    """
         Input a file, specify its type, and click on the button to generate a report.
         """
)

# File uploader
uploaded_file = st.file_uploader("Upload a file")

if uploaded_file is not None:
    st.divider()

    # Select file extension
    left, right = st.columns(2)
    type_file = left.selectbox(
        label="Select a file extension", options=["csv"]
    )

    # Generate report button
    if right.button("Generate Report"):
        st.divider()
        st.markdown(f"### Report for *{uploaded_file.name}*")
        clean_dataset(uploaded_file, type_file)
