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

    # If ID in columns set type to str
    if "ID" in df.columns:
        df["ID"] = df["ID"].astype(str)

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

    for var in var_kept:
        df_tmp = df.dropna(subset=[var])
        print(f"This includes dropping {tmp_len - len(df_tmp)} entries with NaN {var}")

    return df_out


def drug_therapy_color_dict():
    return {
        "Trikafta": "green",
        "Ivacaftor": "purple",
        "Symkevi": "purple",
        "Orkambi": "purple",
    }


def add_drug_therapy_shapes_for_ID(fig, df_for_ID, drug_df):
    drug_df = drug_df[drug_df["DrugTherapyType"] != "Unknown"]
    drug_df_for_ID = drug_df[drug_df.ID == df_for_ID.ID[0]]
    for _, row in drug_df_for_ID.iterrows():
        start_date = row.DrugTherapyStartDate
        stop_date = row.DrugTherapyStopDate
        if pd.isnull(stop_date):
            stop_date = df_for_ID["Date Recorded"].max()

        fig.add_shape(
            dict(
                type="rect",
                xref="x",
                # yref="y",
                yref="paper",
                x0=start_date,
                y0=0,
                x1=stop_date,
                y1=1,
                fillcolor=drug_therapy_color_dict()[row.DrugTherapyType],
                opacity=0.08,
                layer="below",
                line_width=0,
                name=row.DrugTherapyType,
                # label=dict(text=row.DrugTherapyType, textposition="top center", font=dict(size=20)),
            )
        )
        # Add annotation
        fig.add_annotation(
            x=start_date,
            y=1.02,
            xref="x",
            yref="paper",
            text=row.DrugTherapyType,
            showarrow=False,
            font=dict(size=10),
        )
    return -1
