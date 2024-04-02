import os

import numpy as np
import pandas as pd


def get_path_to_src():
    return os.getcwd().split("src")[0] + "src/"


def get_path_to_main():
    return os.getcwd().split("PhD")[0] + "PhD/"


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

    # Convert the given string columns to arrays
    if str_cols_to_arrays:
        for col in str_cols_to_arrays:
            df[col] = df[col].apply(_str_to_array)
    return df


def _str_to_array(s):
    s_cleaned = s.replace("\\n", "")
    return np.fromstring(s_cleaned[1:-1], sep=" ")


def remove_any_nan(df, var_kept):
    """
    Removes entries with NaN in any of the variables in var_kept
    """
    tmp_len = len(df)
    df_out = df.dropna(subset=var_kept, how="any")
    print(
        f"Dropped {tmp_len - len(df_out)} entries with at least one NaN in subset {var_kept}"
    )
    print(f"{len(df)}/{tmp_len} entries remain")

    for var in var_kept:
        df_tmp = df.dropna(subset=[var])
        print(f"This includes dropping {tmp_len - len(df_tmp)} entries with NaN {var}")

    return df_out
