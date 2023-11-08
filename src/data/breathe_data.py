import biology as bio
import numpy as np
import pandas as pd
import sanity_checks


def load_patients():
    """
    Loads Breathe patient data from the excel file and returns a dataframe
    Columns loaded: ID, Age, Sex, Height
    """
    print("\n*** Loading patients data ***")
    df = pd.read_excel(
        "../../../../DataFiles/BR/PredModInputData.xlsx",
        sheet_name="brPatient",
        usecols="A, J, K, L",
    )
    # Set ID as string
    df.ID = df.ID.astype(str)

    sanity_checks.data_types(df)
    df.apply(lambda x: sanity_checks.age(x.Age, x.ID), axis=1)
    df.apply(lambda x: sanity_checks.sex(x.Sex, x.ID), axis=1)
    df.apply(lambda x: sanity_checks.height(x.Height, x.ID), axis=1)

    print(f"Loaded {len(df)} individuals")
    return df


def load_measurements():
    """
    Loads the Breathe data from the excel file and returns a dataframe
    Only loads FEV1 and O2 Saturation measurements
    """
    print("\n*** Loading measurements data ***")
    df = pd.read_excel(
        "../../../../DataFiles/BR/PredModInputData.xlsx",
        sheet_name="BRphysdata",
        usecols="A, E, G, H , J",
    )

    df = df.drop(["RecordingType"], axis=1)
    df.rename(
        columns={
            "FEV": "FEV1",
            "O2Saturation": "O2 Saturation",
            "SmartCareID": "ID",
            "Date_TimeRecorded": "DateTime Recorded",
        },
        inplace=True,
    )
    df = df.astype(
        {
            "ID": str,
            "FEV1": float,
            "DateTime Recorded": "datetime64[ns]",
        }
    )
    df["Date Recorded"] = df["DateTime Recorded"].dt.date

    df = _correct_o2_saturation(df)
    sanity_checks.data_types(df)

    # Apply sanity checks on O2 Saturation and FEV1
    # df.apply(lambda x: sanity_checks.fev1(x["FEV1"], x.ID), axis=1)
    # df.apply(lambda x: sanity_checks.o2_saturation(x["O2 Saturation"], x.ID), axis=1)

    return df


def _correct_o2_saturation(df):
    """
    Corrects O2 Saturation values
    """
    # Round all values to the integer
    df["O2 Saturation"] = df["O2 Saturation"].round().astype(int)
    return df


def build_O2_FEV1_df():
    """
    Merges patients and measurement dataframes
    Computes base variables Predicted FEV1, FEV1 % Predicted
    Computes additional variables: Healthy O2 Saturation, Avg FEV1 % Predicted, Avg Predicted FEV1
    """
    print("\n*** Building O2 Saturation and FEV1 dataframe ***")
    df_patients = load_patients()
    df_meas = load_measurements()

    # Merge rows with same SmartCareId and DateRecorded, taking the non NaN value
    ## Define custom aggregation function
    def custom_aggregation(series):
        non_nan_values = series.dropna()
        if len(non_nan_values) > 1:
            raise ValueError(
                f"More than 1 non NaN value in group: {non_nan_values.tolist()}"
            )
        if len(non_nan_values) == 0:
            return np.nan
        print(f"non_nan_values: {non_nan_values}")
        return non_nan_values.iloc[0]

    df_meas = df_meas.groupby(["SmartCareID", "DateRecorded"])[
        ["FEV1", "O2 Saturation"]
    ].agg(custom_aggregation)

    # Count rows where FEV and O2 Saturation are NaN together
    print("FEV1 and O2 Saturation NaN together: ", df_meas.isna().all(axis=1).sum())
    # Count and print rows where either FEV or O2 Saturation is NaN
    print("FEV1 or O2 Saturation is NaN: ", df_meas.isna().any(axis=1).sum())
    # Print number of rows
    print("Number of rows: ", len(df_meas))
    # Drop rows with nan values
    df_meas = df_meas.dropna()
    print("Dropping NaN rows")
    # Count number of rows
    print("Number of rows: ", len(df_meas))

    # Merge patient and measurement dataframes on SmartCareID and ID
    df = df_meas.merge(df_patients, right_on="ID", left_on="SmartCareID", how="left")

    # Print number of IDs
    print(
        "Number of IDs, datapoints after merging patient and measurement data: ",
        df.ID.nunique(),
        len(df),
    )

    # Compute predicted FEV1 using calc_predicted FEV1 in the biology module
    df["Predicted FEV1"] = df.apply(
        lambda x: bio.calc_LMS_predicted_FEV1(
            bio.load_LMS_spline_vals(x.Age, x.Sex),
            bio.load_LMS_coeffs(x.Sex),
            x.Height,
            x.Age,
            x.Sex,
        )["mean"],
        axis=1,
    )
    # Compute FEV1 % Predicted
    df["FEV1 % Predicted"] = df["FEV1"] / df["Predicted FEV1"] * 100

    # Compute avg FEV1 % Predicted per individual
    def compute_avg(df, col_name, unit):
        tmp = df.groupby("ID")[col_name].mean()
        # Add tmp to a new column per Id
        df = df.join(tmp, on="ID", rsuffix="_avg")

        df[f"ID (avg {col_name})"] = df.apply(
            lambda x: f"{x.ID} ({str(round(x[f'{col_name}_avg'],1))}{unit})",
            axis=1,
        )
        return df

    df = compute_avg(df, "FEV1 % Predicted", "%")
    df = compute_avg(df, "FEV1", "L")

    df[f"ID (Predicted FEV1)"] = df.apply(
        lambda x: f"{x.ID} ({str(round(x['Predicted FEV1'],1))}L)",
        axis=1,
    )

    df["Healthy O2 Saturation"] = df.apply(
        lambda x: bio.calc_healthy_O2_saturation(x["O2 Saturation"], x.Sex, x.Height)[
            "mean"
        ],
        axis=1,
    )

    return df
