import logging
from functools import reduce

import pandas as pd

import src.data.helpers as dh
import src.data.sanity_checks as sanity_checks
import src.modelling_fev1.ec_smoothing as ec_smoothing
import src.modelling_fev1.pred_fev1 as pred_fev1
import src.modelling_o2.ho2sat as ho2sat


def load_meas_from_excel(filename, str_cols_to_arrays=None):
    df = pd.read_excel(dh.get_path_to_main() + f"ExcelFiles/BR/{filename}.xlsx")
    # ID column as type string
    df["ID"] = df["ID"].astype(str)
    # Date Redocrded as datetime
    df["Date Recorded"] = df["Date Recorded"].dt.date

    # Convert the given string columns to arrays
    if str_cols_to_arrays:
        for col in str_cols_to_arrays:
            df[col] = df[col].apply(dh._str_to_array)
    return df


def load_patients():
    """
    Loads Breathe patient data from the excel file and returns a dataframe
    Columns loaded: ID, DOB, Age, Sex, Height
    """
    logging.info("*** Loading patients data ***")

    df = pd.read_csv(
        f"{dh.get_path_to_main()}DataFiles/BR/REDCapData/ProcessedData/brPatient_20240510.csv",
        # usecols="A, E, G, H, Y"
        usecols=[0, 6, 7, 24],
        dtype={"ID": str},
    )
    # Use calc age insted of age
    df = df.rename(columns={"CalcAge": "Age"})

    sanity_checks.data_types(df)
    sanity_checks.must_not_have_nan(df)

    def patients_sanity_checks(x):
        sanity_checks.age(x.Age, x.ID)
        sanity_checks.sex(x.Sex, x.ID)
        sanity_checks.height(x.Height, x.ID)

    df.apply(patients_sanity_checks, axis=1)

    logging.info(f"Loaded {len(df)} individuals")
    describe_patients(df)
    return df


def describe_patients(df):
    """
    Describes the patients dataframe
    """
    logging.info(f"IDs: {df.ID.nunique()}")
    logging.info(f"Sex: {df.Sex.value_counts()}")
    logging.info(
        f"Age (yr): min {df.Age.min()}, mean {df.Age.mean():.1f}, max {df.Age.max()}, std {df.Age.std():.1f}"
    )
    logging.info(
        f"Height (cm): min {df.Height.min()}, mean {df.Height.mean():.1f}, max {df.Height.max()}, std {df.Height.std():.1f}"
    )


def load_measurements(file=2, fef2575=True):
    """
    Loads the Breathe data from the excel file and returns a dataframe
    Only loads FEV1 and O2 Saturation measurements
    """
    print("\n*** Loading measurements data ***")
    if file == 1:
        df_raw = (
            pd.read_excel(
                f"{dh.get_path_to_main()}DataFiles/BR/PredModInputData.xlsx",
                sheet_name="BRphysdata",
                usecols="A, E, G, H , J",
            )
            .rename(
                columns={"SmartCareID": "ID", "Date_TimeRecorded": "DateTime Recorded"}
            )
            .astype({"ID": str, "DateTime Recorded": "datetime64[ns]"})
        )
    elif file == 2:
        df_raw = (
            pd.read_excel(
                f"{dh.get_path_to_main()}DataFiles/BR/MeasurementData/ProcessedData/BRPhysdata-20240510T160334.xlsx",
                usecols="A, E, G, H , J",
            )
            .rename(
                columns={"SmartCareID": "ID", "Date_TimeRecorded": "DateTime Recorded"}
            )
            .astype({"ID": str, "DateTime Recorded": "datetime64[ns]"})
        )

    df_raw["Date Recorded"] = df_raw["DateTime Recorded"].dt.date

    # Drop DateTime Recorded column
    df_raw.drop(columns=["DateTime Recorded"], inplace=True)

    # Process subset of measures
    df_meas_list = []

    # FEV1
    df_fev1 = _get_measure_from_raw_df(
        df_raw, "FEV", "FEV1Recording", type=float, new_col_name="FEV1"
    )
    df_fev1 = _correct_fev1(df_fev1)
    sanity_checks.same_day_measurements(df_fev1)
    df_fev1.apply(lambda x: sanity_checks.fev1(x["FEV1"], x.ID), axis=1)
    df_meas_list.append(df_fev1)

    # O2 saturation
    df_o2_sat = _get_measure_from_raw_df(
        df_raw,
        "O2Saturation",
        "O2SaturationRecording",
        # TODO: Why can O2 Saturation be a float?
        type=float,
        new_col_name="O2 Saturation",
    )
    sanity_checks.same_day_measurements(df_o2_sat)
    df_o2_sat.apply(
        lambda x: sanity_checks.o2_saturation(x["O2 Saturation"], x.ID), axis=1
    )
    df_meas_list.append(df_o2_sat)

    # FEF2575
    if fef2575:
        df_fef2575 = _get_measure_from_raw_df(
            df_raw, "FEV", "FEF2575Recording", type=float, new_col_name="FEF2575"
        )
        sanity_checks.same_day_measurements(df_fef2575)
        df_fef2575.apply(lambda x: sanity_checks.fef2575(x["FEF2575"], x.ID), axis=1)
        df_meas_list.append(df_fef2575)

        df_pef = _get_measure_from_raw_df(
            df_raw, "FEV", "PEFRecording", type=int, new_col_name="PEF"
        )
        sanity_checks.same_day_measurements(df_pef)
        df_pef.apply(lambda x: sanity_checks.pef(x["PEF"], x.ID), axis=1)
        df_meas_list.append(df_pef)

    # Build final dataframe, must have at least 2 elements in list
    df_meas = reduce(
        lambda left, right: pd.merge(
            left, right, on=["ID", "Date Recorded"], how="outer"
        ),
        df_meas_list,
    )

    sanity_checks.data_types(df_meas)

    print("Number of IDs: ", df_meas.ID.nunique())
    print("Number of rows: ", len(df_meas))
    print(f"Number of FEV1 recordings: {len(df_fev1)}")
    if fef2575:
        print(f"Number of FEF2575 recordings: {len(df_fef2575)}")
        print(f"Number of PEF recordings: {len(df_pef)}")
    print(f"Number of O2 Saturation recordings: {len(df_o2_sat)}")
    return df_meas


def _get_measure_from_raw_df(
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
    # UPDATE: one and only measurement for that ID
    df = _remove_recording(df, "330", "FEV1", 6.0)
    return df


def build_O2_FEV1_df(meas_file=2):
    """
    Drop NaN entries
    Merges patients and measurement dataframes
    Computes the following additional variables: Predicted FEV1, FEV1 % Predicted, Healthy O2 Saturation, Avg FEV1 % Predicted, Avg Predicted FEV1
    """
    var_kept = ["O2 Saturation", "FEV1"]
    print("\n*** Building O2 Saturation and FEV1 dataframe ***")

    df_patients = load_patients()
    df_meas = load_measurements(file=meas_file, fef2575=False)
    df_meas = dh.remove_any_nan(df_meas, var_kept)

    df_meas = ec_smoothing.calc_smoothed_fe_measures(df_meas)

    df_patients = calc_predicted_FEV1_LMS_df(df_patients)
    df_patients = calc_healthy_O2_sat_df(df_patients)

    df = df_meas.merge(df_patients, on="ID", how="left")

    df = calc_FEV1_prct_predicted_df(df)
    df = calc_O2_sat_prct_healthy_df(df)

    print(f"Built data structure with {df.ID.nunique()} IDs and {len(df)} entries")

    return df


def build_O2_FEV1_FEF2575_PEF_df(remove_nan, meas_file=2):
    """
    Drop NaN entries
    Merges patients and measurement dataframes
    Computes the following additional variables: Predicted FEV1, FEV1 % Predicted, Healthy O2 Saturation, Avg FEV1 % Predicted, Avg Predicted FEV1
    """
    var_kept = ["O2 Saturation", "FEV1", "FEF2575", "PEF"]
    print("\n*** Building O2Sat, FEV1, FEF2575 dataframe ***")

    df_patients = load_patients()
    df_meas = load_measurements(file=meas_file)
    df_meas["PEF (L/s)"] = df_meas.PEF / 60

    if remove_nan:
        df_meas = dh.remove_any_nan(df_meas, var_kept)

    df_meas = ec_smoothing.calc_smoothed_fe_measures(df_meas)

    df_patients = calc_predicted_FEV1_LMS_df(df_patients)
    df_patients = calc_healthy_O2_sat_df(df_patients)

    df = df_meas.merge(df_patients, on="ID", how="left")

    df = calc_FEV1_prct_predicted_df(df)
    df = calc_O2_sat_prct_healthy_df(df)

    print(f"Built data structure with {df.ID.nunique()} IDs and {len(df)} entries")

    return df


def calc_predicted_FEV1_LMS_df(df):
    """
    Returns a Series with Predicted FEV1 from a DataFrame with Sex, Height, Age
    """
    df["Predicted FEV1"] = df.apply(
        lambda x: pred_fev1.calc_predicted_value_LMS_straight(
            x.Height,
            x.Age,
            x.Sex,
        )["M"],
        axis=1,
    )
    df.apply(lambda x: sanity_checks.predicted_fev1(x["Predicted FEV1"], x.ID), axis=1)
    return df


def calc_FEV1_prct_predicted_df(df):
    """
    Returns input DataFrame with FEV1 % Predicted as a new column, after sanity check
    """
    df["ecFEV1 % Predicted"] = df["ecFEV1"] / df["Predicted FEV1"] * 100
    df.apply(
        lambda x: sanity_checks.fev1_prct_predicted(x["ecFEV1 % Predicted"], x.ID),
        axis=1,
    )
    df["FEV1 % Predicted"] = df["FEV1"] / df["Predicted FEV1"] * 100
    df.apply(
        lambda x: sanity_checks.fev1_prct_predicted(x["FEV1 % Predicted"], x.ID), axis=1
    )
    return df


def calc_healthy_O2_sat_df(df):
    """
    Returns input DataFrame with added column Healthy O2 Saturation, given Height and Sex
    """
    df["Healthy O2 Saturation"] = df.apply(
        lambda x: ho2sat.calc_healthy_O2_sat(x.Height, x.Sex)["mean"],
        axis=1,
    )

    return df


def calc_O2_sat_prct_healthy_df(df):
    """
    Returns input DataFramce with added column O2 Saturation % Healthy, given O2 Saturation and Healthy O2 Saturation
    """
    df["O2 Saturation % Healthy"] = (
        df["O2 Saturation"] / df["Healthy O2 Saturation"] * 100
    )
    return df
