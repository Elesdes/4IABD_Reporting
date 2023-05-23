import csv
import io
from typing import Any, Optional

import pandas as pd
import streamlit as st


def get_delimiter(metadata: Any) -> Optional[str]:
    """
    This function takes a metadata object and returns the delimiter of a CSV file.
    If the delimiter cannot be determined, it returns None.

    Args:
        metadata (Any): The metadata object.

    Returns:
        Optional[str]: The delimiter of the CSV file or None if the delimiter cannot be determined.
    """
    string_io = io.StringIO(metadata.getvalue().decode("utf-8"))
    sniffer = csv.Sniffer()
    dialect = sniffer.sniff(string_io.read(1024))
    return dialect.delimiter


def generate_df_from_metadata(metadata: Any, file_type: str) -> Optional[pd.DataFrame]:
    """
    Generates a Pandas DataFrame from metadata of a given file type.

    Args:
        metadata (Any): The metadata of the file.
        file_type (str): The type of the file (e.g. "csv", "xlsx", "txt").

    Returns:
        Optional[pd.DataFrame]: The DataFrame generated from the file metadata, or None if an error occurred.
    """
    try:
        match file_type if metadata.name.endswith(file_type) else None:
            case "csv":
                delimiter = get_delimiter(metadata)
                df = pd.read_csv(metadata, delimiter=delimiter)
            case "xlsx":
                df = pd.read_excel(metadata)
            case "txt":
                df = pd.read_csv(metadata, sep="\t")
            case _:
                st.error(f"Incorrect file extension. Expected '{file_type}'.")
                return None
    except Exception as e:
        st.error(f"An error occurred while reading the file: '{e}'")
        return None
    return df
