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


def load_demographics_data(
    cols2read=["s01caseid_original", "s01sex"], colnames=["ID", "Sex"]
):
    df_demo = pd.read_excel(
        dh.get_path_to_main() + f"DataFiles/CFR/CF_Registry_2019_23_Floto_Output.xlsx",
        sheet_name="Demographics",
        usecols=cols2read,
    )

    rename_dict = dict(zip(cols2read, colnames))
    df_demo = df_demo.rename(columns=rename_dict)

    def get_back_sex(sex):
        if sex == "F":
            return "Female"
        elif sex == "M":
            return "Male"
        else:
            logging.warning(f"Sex: Error converting {sex} to string")
            return None

    df_demo.Sex = df_demo.Sex.apply(get_back_sex)

    return df_demo


def build_cfr_df(
    year,
    cols2read=[
        "s01caseid_original",
        # "s01sex",
        "s01height",
        "s01encounterageyears",
        "s03cliqtrfev1",  # Value at annual review
        # "s03clibestfev1",
        "s03clifef2575",  # Value at annual review
    ],
    colnames=["ID", "Height", "Age", "FEV1", "FEF2575"],
    demo2read=["s01caseid_original", "s01sex"],
    demonames=["ID", "Sex"],
    bypass_sanity_checks=False,
):
    """
    Build CF registry
    """

    df_meas = load_cfr_data(year, cols2read, colnames)
    df_demo = load_demographics_data(demo2read, demonames)

    df = df_meas.merge(df_demo, on=["ID"])

    # Drop NaN
    logging.info(f"Loaded {df.shape[0]} entries")
    df = df.dropna()
    logging.info(f"{df.shape[0]} after removing all NaN")

    # Format to known colnames
    if not bypass_sanity_checks:
        sanity_checks.data_types(df)

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
