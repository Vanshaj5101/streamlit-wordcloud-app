import pandas as pd
import io


def load_csv(uploaded_file):
    """
    Reads a CSV file from a Streamlit uploader.

    Args:
        uploaded_file (UploadedFile): File object from Streamlit uploader.

    Returns:
        pd.DataFrame: Loaded data as a DataFrame.
    """
    return pd.read_csv(uploaded_file)
