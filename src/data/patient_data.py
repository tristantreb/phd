import biology
import pandas as pd
import sanity_checks
from dateutil.relativedelta import relativedelta

datadir = "../../../../SmartCareData/"


def load(use_calc_age=True, use_calc_predicted_fev1=True):
    print("\n** Loading patient data **")
    df = pd.read_excel(
        datadir + "clinicaldata_updated.xlsx", sheet_name="Patients", dtype={"ID": str}
    )
    n_initial_entries = df.shape[0]

    # Find duplicated IDs
    assert df[list(df.ID.value_counts() > 1)].empty, "Duplicated IDs found"

    # Drop columns that are not needed
    # List of columns to keep
    tmp_columns = df.columns
    columns_to_keep = [
        "ID",
        "Study Date",
        "DOB",
        "Age",
        "Sex",
        "Height",
        "Weight",
        "Predicted FEV1",
        "FEV1 Set As",
    ]
    df = df[columns_to_keep]
    print("\n* Dropping unnecessary columns from patient data *")
    print("Columns filtered: {}".format(columns_to_keep))
    print("Columns dropped: {}".format(set(tmp_columns) - set(columns_to_keep)))

    # Clean data types
    df.Weight = df.Weight.replace(to_replace="75,4", value="75.4")

    # Enforce data types
    df = df.astype(
        {
            "ID": str,
            "Age": int,
            "Sex": str,
            "Height": float,
            "Weight": float,
            "Predicted FEV1": float,
            "FEV1 Set As": float,
        }
    )
    # Cast datetime to date
    df["Study Date"] = pd.to_datetime(df["Study Date"]).dt.date
    df.DOB = pd.to_datetime(df.DOB).dt.date

    # Correct erroneous data
    df = _correct_df(df)

    # Compute age and predicted FEV1 if necessary
    if use_calc_age:
        print("Replace Age by calculate age")
        df.Age = df.apply(
            lambda row: round(_get_years_decimal_delta(row.DOB, row["Study Date"])),
            axis=1,
        )
    if use_calc_predicted_fev1:
        print("Drop FEV1 Set As and Predicted FEV1")
        df = df.drop(columns=["FEV1 Set As", "Predicted FEV1"])
        df = _compute_predicted_fev1(df)

    # Apply data sanity checks
    print("\n* Applying data sanity checks *")
    df.apply(_age_sanity_check, axis=1)
    df.apply(_sex_sanity_check, axis=1)
    df.apply(_height_sanity_check, axis=1)
    df.apply(lambda x: sanity_checks.weight(x.Weight, x.ID), axis=1)
    df.apply(_predicted_fev1_sanity_check, axis=1)

    print(
        "Loaded patient data with {} entries ({} initially)".format(
            df.shape[0], n_initial_entries
        )
    )
    return df


def _age_sanity_check(row):
    assert (
        row.Age >= 18 and row.Age <= 70
    ), "Warning - ID {} has Age ({}) outside 18-70 range".format(row.ID, row.Age)
    return -1


def _sex_sanity_check(row):
    assert row.Sex in [
        "Male",
        "Female",
    ], "Sex ({}) is not 'Male' neither 'Female'".format(row.ID, row.Sex)
    return -1


def _height_sanity_check(row):
    assert (
        row.Height >= 120 and row.Height <= 220
    ), "Warning - ID {} has Height ({}) outside 120-220cm range".format(
        row.ID, row.Height
    )
    return -1


def _predicted_fev1_sanity_check(row):
    if row["Predicted FEV1"] >= 5 or row["Predicted FEV1"] <= 2:
        print(
            "Warning - ID {} has Predicted FEV1 ({}) outside 2-5L range".format(
                row.ID, row["Predicted FEV1"]
            )
        )
    return -1


def _correct_df(df):
    print("\n* Correcting patient data *")
    # ID 60, convert height from m to cm
    tmp = df.loc[df.ID == "60", "Height"]
    df.Height.loc[df.ID == "60"] = tmp * 100

    # Print data correction for ID 60
    print(
        "ID 60: Corrected height 60 from {} to {}".format(
            (tmp).to_string(index=False),
            (tmp * 100).to_string(index=False),
        )
    )
    # ID 66, convert height from m to cm
    tmp = df.loc[df.ID == "66", "Height"]
    df.loc[df.ID == "66", "Height"] = tmp * 100

    # Print data correction for ID 66
    print(
        "ID 66: Corrected height for ID 66 from {} to {}".format(
            (tmp).to_string(index=False),
            (tmp * 100).to_string(index=False),
        )
    )
    return df


def _get_years_decimal_delta(start_date, end_date):
    return (
        relativedelta(end_date, start_date).years
        + relativedelta(end_date, start_date).months / 12
    )


def _compute_predicted_fev1(df):
    """
    Compute Predicted FEV1
    Checks that Predicted FEV1 is always in the range 2-5.5 L
    """
    print("Compute Calculated Predicted FEV1")
    df["Predicted FEV1"] = df.apply(
        lambda x: biology.calc_predicted_fev1(x.Height, x.Age, x.Sex), axis=1
    )
    # Assert type is float
    assert df["Predicted FEV1"].dtype == float
    # Assert Predicted FEV1 is always in the range 2-5.5
    assert df["Predicted FEV1"].min() >= 2
    assert df["Predicted FEV1"].max() <= 5.5
    return df
