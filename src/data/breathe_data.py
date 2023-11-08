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
    sanity_checks.must_not_have_nan(df)
    print(
        "The 4 NaN values belong to IDs ('322', '338', '344', '348') whose height are missing.\nHowever, we don't correct for them as we don't have any measurement corresponding to those IDs for now."
    )

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
    df_raw = (
        pd.read_excel(
            "../../../../DataFiles/BR/PredModInputData.xlsx",
            sheet_name="BRphysdata",
            usecols="A, E, G, H , J",
        )
        .rename(columns={"SmartCareID": "ID", "Date_TimeRecorded": "DateTime Recorded"})
        .astype({"ID": str, "DateTime Recorded": "datetime64[ns]"})
    )

    df_raw["Date Recorded"] = df_raw["DateTime Recorded"].dt.date

    # Drop DateTime Recorded column
    df_raw.drop(columns=["DateTime Recorded"], inplace=True)

    df_fev1 = get_measure_from_raw_df(
        df_raw, "FEV", "FEV1Recording", type=float, new_col_name="FEV1"
    )
    df_fev1 = _correct_fev1(df_fev1)
    sanity_checks.same_day_measurements(df_fev1, "FEV1")
    df_fev1.apply(lambda x: sanity_checks.fev1(x["FEV1"], x.ID), axis=1)

    df_o2_sat = get_measure_from_raw_df(
        df_raw,
        "O2Saturation",
        "O2SaturationRecording",
        # TODO: Why can O2 Saturation be a float?
        type=float,
        new_col_name="O2 Saturation",
    )
    sanity_checks.same_day_measurements(df_o2_sat, "O2 Saturation")
    df_o2_sat.apply(
        lambda x: sanity_checks.o2_saturation(x["O2 Saturation"], x.ID), axis=1
    )

    df_meas = pd.merge(df_fev1, df_o2_sat, on=["ID", "Date Recorded"], how="outer")

    sanity_checks.data_types(df_meas)

    print("Number of IDs: ", df_meas.ID.nunique())
    print("Number of rows: ", len(df_meas))
    print(f"Number of FEV1 recordings: {len(df_fev1)}")
    print(f"Number of O2 Saturation recordings: {len(df_o2_sat)}")
    return df_meas


def get_measure_from_raw_df(
    df, col_name, recording_type, type=False, new_col_name=False
):
    df = df[df.RecordingType == recording_type]
    df = df[["ID", "Date Recorded", col_name]]
    if type:
        df = df.astype({col_name: type})
    if new_col_name:
        df.rename(columns={col_name: new_col_name}, inplace=True)
    return df


def _remove_recording(df, id, column, value):
    """
    Removes a recording from a dataframe
    """
    idx = df[(df.ID == id) & (df[column] == value)].index
    print(
        "Dropping {} entries with {} = {} for ID {}".format(
            idx.shape[0], column, value, id
        )
    )
    df.drop(idx, inplace=True)
    return df


def _correct_fev1(df):
    """
    Corrects FEV1 values
    """
    # Warning - ID 330 has FEV1 (6.0) outside 0.1-5.5 L range
    # Corresponds to 24y, Female, 153.5cm -> predicted FEV1: 2.9
    # Let's remove this entry
    df = _remove_recording(df, "330", "FEV1", 6.0)
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
