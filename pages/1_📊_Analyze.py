import pandas as pd
import streamlit as st
from src.analyze_marijuana import convert_data, render
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="Marijuana Arrest In Colombia - Analyze", page_icon="chart_with_upwards_trend")

# Add title and description
st.title("Marijuana Arrest In Colombia")
st.sidebar.header("Discover Our Project.")
st.write(
    """
         Here is our analyze.
         """
)


data = pd.read_csv("data/Marijuana_Arrests.csv")
columns = data.columns

for column in columns:
    dataset, data_type = convert_data(
        data[column]
    )  # Convert data to numerical, categorical, text or index

    dataset = dataset.dropna()  # Remove NaN values

    sparsity = 1.0 - len(dataset) / float(
        len(data[column])
    )  # 1 - Size after cleaning / Size before cleaning

    render(dataset, column, sparsity, data_type)
