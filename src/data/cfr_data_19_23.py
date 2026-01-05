import logging

import pandas as pd

import data.breathe_data as bd
import data.helpers as dh
import data.sanity_checks as sanity_checks


def load_cfr_data(year, cols2read, colnames):
    sheet_name = f"{year} Annual Review"

    df = pd.read_excel(
        dh.get_path_to_main() + f"DataFiles/CFR/CF_Registry_2019_23_Floto_Output.xlsx",
        sheet_name=sheet_name,
        usecols=cols2read,
    )

    rename_dict = dict(zip(cols2read, colnames))
    df = df.rename(columns=rename_dict)

    return df


def load_demographics_data():
    # Demographics
    cols2read = ["s01caseid_original", "s01sex"]

    df_demo = pd.read_excel(
        dh.get_path_to_main() + f"DataFiles/CFR/CF_Registry_2019_23_Floto_Output.xlsx",
        sheet_name="Demographics",
        usecols=cols2read,
    )
    return df_demo


def build_cfr_df(year):
    """
    Build CF registry
    """
    cols2read = [
        "s01caseid_original",
        # "s01sex",
        "s01height",
        "s01encounterageyears",
        "s03cliqtrfev1",  # Value at annual review
        # "s03clibestfev1",
        "s03clifef2575",  # Value at annual review
    ]
    df_meas = load_cfr_data(year, cols2read)
    df_demo = load_demographics_data()
    df = df_meas.merge(df_demo, on=["s01caseid_original"])

    # Drop NaN
    logging.info(f"Loaded {df.shape[0]} entries")
    df = df.dropna()
    logging.info(f"{df.shape[0]} after removing all NaN")

    # Format to known colnames
    df = df.rename(
        columns={
            "s01caseid_original": "ID",
            "s01encounterageyears": "Age",
            "s01sex": "Sex",
            "s01height": "Height",
            "s03cliqtrfev1": "FEV1",
            "s03clifef2575": "FEF2575",
        }
    )
    sanity_checks.data_types(df)

    df.Sex = df.Sex.apply(lambda row: "Female" if "F" else "Male")
    df.Height = df.Height.round()
    df["Date Recorded"] = f"{year}-01-01"
    df["Date Recorded"] = pd.to_datetime(df["Date Recorded"]).dt.date

    # Remove < 18yr
    logging.info(f"{(df.Age >= 18).sum()} entries after removing <18yr")
    df = df[df.Age >= 18]

    # Effort correction not applicable
    df["ecFEV1"] = df["FEV1"]
    df["ecFEF2575"] = df["FEF2575"]
    df["ecFEF2575%ecFEV1"] = df["FEF2575"] / df.FEV1 * 100

    # Compute predicted values
    df = bd.calc_predicted_FEV1_LMS_df(df)
    df = bd.calc_FEV1_prct_predicted_df(df)

    return df
