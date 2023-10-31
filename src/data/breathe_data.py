import biology as bio
import numpy as np
import pandas as pd


def load_patients():
    print("*** Loading patients data ***")
    df_patients = pd.read_excel(
        "../../../../DataFiles/BR/PredModInputData.xlsx",
        sheet_name="brPatient",
        usecols="A, J, K, L",
    )
    # Set ID as string
    df_patients["ID"] = df_patients["ID"].astype(str)

    # Print number of patients loaded
    print("Number of IDs: ", len(df_patients))
    return df_patients


def load_measurements():
    """
    Loads the Breathe data from the excel file and returns a dataframe
    Only loads FEV1 and O2 Saturation measurements
    """
    print("*** Loading measurements data ***")
    df_meas = pd.read_excel(
        "../../../../DataFiles/BR/PredModInputData.xlsx",
        sheet_name="BRphysdata",
        usecols="A, E, G, H , J",
    )

    # Set SmartCareID as type string
    df_meas["SmartCareID"] = df_meas["SmartCareID"].astype(str)
    # Rename FEV to FEV1
    df_meas.rename(
        columns={"FEV": "FEV1", "O2Saturation": "O2 Saturation"}, inplace=True
    )

    # Drop rows where Recording Type is not "FEV1Recording", or "O2 SaturationRecording"
    df_meas = df_meas[
        df_meas["RecordingType"].isin(["FEV1Recording", "O2SaturationRecording"])
    ]

    # Replace 0.00 with NaN
    df_meas = df_meas.replace(0.00, np.nan)
    # Create Date Recorded column and drop time from Date/Time Recorded column
    df_meas["DateRecorded"] = df_meas["Date_TimeRecorded"].dt.date
    # Drop Date_TimeRecorded column and RecordingType column
    df_meas = df_meas.drop(["Date_TimeRecorded", "RecordingType"], axis=1)

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
        return non_nan_values.iloc[0]

    df_meas = df_meas.groupby(["SmartCareID", "DateRecorded"])[
        ["FEV1", "O2 Saturation"]
    ].agg(custom_aggregation)

    # Count rows where FEV and O2 Saturation are NaN together
    print("FEV1 and SpO2 NaN together: ", df_meas.isna().all(axis=1).sum())
    # Count and print rows where either FEV or O2 Saturation is NaN
    print("FEV1 or SpO2 is NaN: ", df_meas.isna().any(axis=1).sum())
    # Print number of rows
    print("Number of rows: ", len(df_meas))
    # Drop rows with nan values
    df_meas = df_meas.dropna()
    print("Dropping NaN rows")
    # Count number of rows
    print("Number of rows: ", len(df_meas))

    return df_meas


def build_O2_FEV1_df():
    """
    Merges patients and measurement dataframes
    Computes base variables Predicted FEV1, FEV1 % Predicted
    Computes additional variables: Predicted SpO2, Avg FEV1 % Predicted, Avg Predicted FEV1
    """
    df_patients = load_patients()
    df_meas = load_measurements()

    # Merge patient and measurement dataframes on SmartCareID and ID
    df = df_meas.merge(df_patients, right_on="ID", left_on="SmartCareID", how="left")

    # Print number of IDs
    print("Number of IDs: ", len(df_patients))

    # Compute predicted FEV1 using calc_predicted FEV1 in the biology module
    # df["Predicted FEV1"] = df.apply(lambda row: bio.calc_predicted_fev1(row.Height, row.Age, row.Sex)["Predicted FEV1"], axis=1)
    df["Predicted FEV1"] = df.apply(
        lambda x: bio.calc_LMS_predicted_FEV1(
            bio.load_LMS_spline_vals(x.Age, x.Sex),
            bio.load_LMS_coeffs(x.Sex),
            x.Height,
            x.Age,
            x.Sex,
        )["Predicted FEV1"],
        axis=1,
    )
    # Compute FEV1 % Predicted
    df["FEV1 % Predicted"] = df["FEV1"] / df["Predicted FEV1"] * 100

    # Remove when there's less than 10 O2 Saturation measurements
    tmp_shape = df.shape[0]
    tmp_ids = df.groupby("ID").size()
    df = df.groupby("ID").filter(lambda x: len(x) >= 10)
    print(
        f"Removed {tmp_shape - df.shape[0]}/{tmp_shape} rows, {tmp_ids.shape[0] - df.groupby('ID').size().shape[0]}/{tmp_ids.shape[0]} patients"
    )

    # Remove values below 85 - concerns one individual (ID 111)
    df = df[df["O2 Saturation"] >= 85]

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

    # Compute calc predicted SpO2
    df["Predicted SpO2"] = df.apply(
        lambda x: bio.calc_predicted_SpO2(x["O2 Saturation"], x.Sex), axis=1
    )

    return df
