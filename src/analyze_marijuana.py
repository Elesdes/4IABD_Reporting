from math import ceil, sqrt
from typing import Any, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from .df_utils import generate_df_from_metadata
from matplotlib.backends.backend_agg import RendererAgg

DATA_TYPE_CATEGORICAL = "CAT"
DATA_TYPE_TEXT = "TXT"
DATA_TYPE_NUMERICAL = "NUM"
DATA_TYPE_INDEX = "INDEX"
DATA_TYPE_DATE = "DATE"
MISSING_VALUES = {"TYPE": [['', '']],
                  "ADULT_JUVENILE": [['', '']],
                  "YEAR": [['', '']],
                  "DATETIME": [['', '']],
                  "CCN": [['', '']],
                  "AGE": [['', '']],
                  "OFFENSE_DISTRICT": [['', '']],
                  "OFFENSE_PSA": [['', '']],
                  "OFFENSE_BLOCKX": [['', '']],
                  "OFFENSE_BLOCKY": [['', '']],
                  "DEFENDANT_PSA": [['Out of State', -1], ['Unknown', -2]],
                  "DEFENDANT_DISTRICT": [['', '']],
                  "RACE": [['', '']],
                  "ETHNICITY": [['', '']],
                  "SEX": [['', '']],
                  "CATEGORY": [['', '']],
                  "DESCRIPTION": [['', '']],
                  "ADDRESS": [['', '']],
                  "ARREST_BLOCKX": [['', '']],
                  "ARREST_BLOCKY": [['', '']],
                  "GIS_ID": [['', '']],
                  "CREATOR": [['', '']],
                  "CREATED": [['', '']],
                  "EDITOR": [['', '']],
                  "EDITED": [['', '']],
                  "OBJECTID": [['', '']],
                  "GLOBALID": [['', '']]}

MAX_CATEGORICAL_VALUES = 32

DISPLAY_VALUES = 5
MAX_PIE_BINS = 10


def filter_data(data, header):
    for head in header:
        for missing_value in MISSING_VALUES[head]:
            data[head] = data[head].replace(missing_value[0], missing_value[1])
    return data


def try_convert(data: pd.Series) -> Any:
    """
    Try to convert a Pandas Series into a list of Int in order to see if it's a real num form.
    :param data: The input Pandas Series containing data to be converted.
    :return: A list of Int that is not received or None.
    """
    tmp = []
    for value in data:
        try:
            tmp.append(int(value))
        except:
            return None
    return tmp


def convert_data(data: pd.Series) -> Tuple[pd.Series, str]:
    """
    Convert the data in the given Pandas Series and determine its data type.

    :param data: The input Pandas Series containing data to be converted.
    :return: A tuple containing the converted data as a Pandas Series and the determined data type as a string.
    """
    num_unique_values = data.nunique()

    int_data = pd.to_numeric(data, errors="coerce", downcast="integer")
    str_data = data.astype(str)
    float_data = pd.to_numeric(str_data.str.replace(",", "."), errors="coerce").round(2)
    date_data = pd.to_datetime(str_data, errors="coerce")

    if num_unique_values == len(data):
        data_type = DATA_TYPE_INDEX
    elif num_unique_values <= MAX_CATEGORICAL_VALUES and (not int_data.isna().any() or not str_data.isna().any()):
        data_type = DATA_TYPE_CATEGORICAL
        # data = int_data.replace(-1, pd.NaT)
    elif not float_data.isna().any() or try_convert(data.dropna()):
        data_type = DATA_TYPE_NUMERICAL
        # data = float_data.replace(-1, pd.NaT)
    elif not date_data.isna().any():
        data_type = DATA_TYPE_DATE
        data = date_data.dt.year
    else:
        data_type = DATA_TYPE_TEXT
        data = data.replace("[!?.]", pd.NaT)

    return data, data_type


def render(dataset: pd.Series, column: str, sparsity: float, data_type: str) -> None:
    st.divider()
    st.write(f"### {column} [{data_type}]")
    left, right = st.columns(2)

    if sparsity > 0:
        right.write(f"Sparsity: {100 * sparsity:.2f}%")
    else:
        right.write("No missing values")
    if data_type == DATA_TYPE_CATEGORICAL or data_type != DATA_TYPE_NUMERICAL:
        right.write(f"Distinct values: {dataset.nunique()}")
    else:
        skew, kurtosis = dataset.skew(), dataset.kurtosis()
        right.write(
            f"""Kurtosis: {kurtosis}
                        \nSkew: {skew}"""
        )

    histogram = dataset.value_counts()

    fig, ax = plt.subplots(figsize=(5, 5))

    if data_type == DATA_TYPE_CATEGORICAL and histogram.size <= MAX_PIE_BINS:
        ax.pie(histogram, labels=histogram.index)
        p = plt.gcf()
        p.gca().add_artist(plt.Circle((0, 0), 0.3, color="white"))
        left.pyplot(fig)
    elif 1000 > histogram.size > 1 and data_type != DATA_TYPE_TEXT:
        left.bar_chart(histogram)
    else:
        left.write("No plot available")


def clean_dataset(metadata: Any, file_type: str) -> None:
    data = generate_df_from_metadata(metadata, file_type)
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
