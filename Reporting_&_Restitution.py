import pandas as pd
import streamlit as st
from src.df_utils import generate_df_from_metadata
from src.reporting import clean_dataset, convert_data, render

# Set page configuration
st.set_page_config(page_title="Welcome !")

st.title("Hello !")
st.sidebar.header("Discover Our Project.")

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