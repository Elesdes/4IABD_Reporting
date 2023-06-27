import pandas as pd
import streamlit as st
from src.df_utils import generate_df_from_metadata
from src.reporting import clean_dataset

# Set page configuration
st.set_page_config(page_title="Welcome !")

st.title("Hello !")
st.sidebar.header("Discover Our Project.")