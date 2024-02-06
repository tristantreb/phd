import numpy as np
import pandas as pd


def compute_avg(df, col_name, unit):
    """
    Compute avg of col_name per individual
    """
    tmp = df.groupby("ID")[col_name].mean()
    # Add tmp to a new column per Id
    df = df.join(tmp, on="ID", rsuffix="_avg")

    df[f"ID (avg {col_name})"] = df.apply(
        lambda x: f"{x.ID} ({str(round(x[f'{col_name}_avg'],1))}{unit})",
        axis=1,
    )
    return df


def load_excel(file_path, str_cols_to_arrays=None):
    """
    Load excel file
    Optionally convert string columns to arrays
    """
    df = pd.read_excel(file_path)

    if str_cols_to_arrays:
        for col in str_cols_to_arrays:
            df[col] = df[col].apply(_str_to_array)
    return df


def _str_to_array(s):
    s_cleaned = s.replace("\\n", "")
    return np.fromstring(s_cleaned[1:-1], sep=" ")
