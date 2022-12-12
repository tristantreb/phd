import pandas as pd
from dateutil.relativedelta import relativedelta

datadir = "../../../../SmartCareData/"


def load(use_calc=True):
    print("\n** Loading patient data **")
    df = pd.read_excel(
        datadir + "clinicaldata_updated.xlsx", sheet_name="Patients", dtype={"ID": str}
    )
    n_initial_entries = df.shape[0]

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
    df = _correct_df(df, use_calc)

    # Apply data sanity checks
    print("\n* Applying data sanity checks *")
    df.apply(_age_sanity_check, axis=1)
    df.apply(_sex_sanity_check, axis=1)
    df.apply(_height_sanity_check, axis=1)
    df.apply(_weight_sanity_check, axis=1)
    # Not added because need to decide which PRedicted FEV1 to use
    # df.apply(_predicted_fev1_sanity_check, axis=1)
    # df.apply(_fev1_set_as_sanity_check, axis=1)

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


def _weight_sanity_check(row):
    assert (
        row.Weight >= 30 and row.Weight <= 120
    ), "Warning - ID {} has Weight ({}) outside 30-120kg range".format(
        row.ID, row.Weight
    )
    return -1


def _correct_df(df, use_calc=True):
    print("\n* Correcting patient data *")

    if use_calc:
        print("Replace Age by use_calculated age")
        df.Age = df.apply(
            lambda row: round(_get_years_decimal_delta(row.DOB, row["Study Date"])),
            axis=1,
        )

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
