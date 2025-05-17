import os

import numpy as np
import pandas as pd


def get_blind_colours():
    """
    #0072b2 000,114,178 blue
    #d55e00 213,094,000 vermilion
    #009e73 000,158,115 bluish-green
    #f0e442 240,228,066 yellow
    #e69f00 230,159,000 orange
    #56b4e9 086,180,233 sky-blue
    #cc79a7 204,121,167 reddish-purple
    """
    return ["#0072b2", "#d55e00", "#009e73", "#f0e442", "#e69f00", "#56b4e9", "#cc79a7"]


def get_path_to_src():
    return os.getcwd().split("src")[0] + "/src/"


def get_path_to_main():
    return os.getcwd().split("PhD")[0] + "/PhD/"


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


def load_excel(file_path, str_cols_to_arrays=None, date_cols=[]):
    """
    Load excel file
    Optionally convert string columns to arrays
    """
    df = pd.read_excel(file_path)

    # If ID in columns set type to str
    if "ID" in df.columns:
        df["ID"] = df["ID"].astype(str)

    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col]).dt.date

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


def find_longest_conseq_sequence(df_for_ID, n_missing_days_allowed=0):

    df_for_ID = df_for_ID.reset_index(drop=True)
    df_for_ID["Prev day"] = df_for_ID["Date Recorded"].shift(1)
    df_for_ID["Days elapsed"] = df_for_ID["Date Recorded"] - df_for_ID["Prev day"]

    # Get first idx where Days elapsed is greater than 1
    idx = df_for_ID[
        df_for_ID["Days elapsed"] > pd.Timedelta(days=n_missing_days_allowed + 1)
    ].index
    # Add the first idx
    idx = idx.insert(0, 0)
    # Add last idx
    idx = idx.insert(len(idx), len(df_for_ID))

    # Make the difference between the idxs
    diff = np.diff(idx)

    # Get the longest series of consecutive measurements
    idx_max_diff = np.argmax(diff)
    start_idx = idx[idx_max_diff]

    end_idx = idx[idx_max_diff + 1]

    df_for_ID[start_idx:end_idx]
    return df_for_ID[start_idx:end_idx], start_idx, end_idx


def split_ID_data_in_groups(df, group_size):
    def split_long_series_in_chunks(df, chunk_size):
        df.reset_index(inplace=True, drop=True)
        id = df.ID.unique()[0]
        df.rename(columns={"ID": "ID_chunks"}, inplace=True)
        df_id = df[df.ID_chunks == id]
        for i in range(0, len(df_id), chunk_size):
            df_id.loc[i : i + chunk_size, "ID_chunks"] = f"{id}_{i//chunk_size}"
        return df_id

    df_new = df.groupby("ID").apply(
        lambda x: split_long_series_in_chunks(x, group_size)
    )
    df_new = (
        df_new.reset_index()
        .drop(columns=["level_1"])
        .rename(columns={"ID": "ID_init", "ID_chunks": "ID"})
    )
    return df_new
