import datetime
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

    df = _remove_patient(df, "344")

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


def _remove_patient(df, id):
    """
    Removes a patient from the patients dataframe
    """
    idx = df[df.ID == id].index
    logging.warning(
        f"ID {id} - Dropping patient because height not referenced (and no measurement to date - 10.05.2024 - for this individual"
    )
    df.drop(idx, inplace=True)
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


def load_measurements(file=2):
    """
    Loads the Breathe data from the excel file and returns a dataframe
    Loads FEV1, O2 saturation, FEF2575 and PEF
    """
    logging.info("*** Loading measurements data ***")
    if file == 1:
        df_raw = pd.read_excel(
            f"{dh.get_path_to_main()}DataFiles/BR/PredModInputData.xlsx",
            sheet_name="BRphysdata",
            usecols="A, E, G, H , J",
            dtype={"SmartCareID": str, "Date_TimeRecorded": "datetime64[ns]"},
        ).rename(
            columns={"SmartCareID": "ID", "Date_TimeRecorded": "DateTime Recorded"}
        )
    elif file == 2:
        df_raw = pd.read_excel(
            f"{dh.get_path_to_main()}DataFiles/BR/MeasurementData/ProcessedData/BRPhysdata-20240510T160334.xlsx",
            usecols="A, E, G, H , J",
            dtype={"SmartCareID": str, "DateTime Recorded": "datetime64[ns]"},
        ).rename(
            columns={"SmartCareID": "ID", "Date_TimeRecorded": "DateTime Recorded"}
        )

    df_raw["Date Recorded"] = df_raw["DateTime Recorded"].dt.date

    # Drop DateTime Recorded column
    df_raw.drop(columns=["DateTime Recorded"], inplace=True)

    # Process subset of measures
    df_meas_list = []

    # FEV1
    logging.info("Processing FEV1")
    df_fev1 = _get_measure_from_raw_df(
        df_raw, "FEV", "FEV1Recording", type=float, new_col_name="FEV1"
    )
    df_fev1 = _correct_fev1(df_fev1)
    sanity_checks.same_day_measurements(df_fev1)
    df_fev1.apply(lambda x: sanity_checks.fev1(x["FEV1"], x.ID), axis=1)
    df_meas_list.append(df_fev1)

    # O2 saturation
    logging.info("Processing O2 saturation")
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
    logging.info("Processing FEF2575")
    df_fef2575 = _get_measure_from_raw_df(
        df_raw, "FEV", "FEF2575Recording", type=float, new_col_name="FEF2575"
    )
    sanity_checks.same_day_measurements(df_fef2575)
    df_fef2575.apply(lambda x: sanity_checks.fef2575(x["FEF2575"], x.ID), axis=1)
    df_meas_list.append(df_fef2575)

    # PEF
    logging.info("Processing PEF")
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

    df_meas.apply(
        lambda x: sanity_checks.fef2575_sup_pef(x["FEF2575"], x["PEF"], x.ID), axis=1
    )

    sanity_checks.data_types(df_meas)

    logging.info(f"Number of IDs: {df_meas.ID.nunique()}")
    logging.info(f"Number of rows: {len(df_meas)}")
    logging.info(f"Number of FEV1 recordings: {len(df_fev1)}")
    logging.info(f"Number of FEF2575 recordings: {len(df_fef2575)}")
    logging.info(f"Number of PEF recordings: {len(df_pef)}")
    logging.info(f"Number of O2 Saturation recordings: {len(df_o2_sat)}")
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


def _remove_recordings_in_date_range(df, id, column, start_date, end_date):
    """
    Removes all recordings in a specific time period (start/end dates included) from a dataframe
    """
    idx = df[
        (df.ID == id)
        & (df["Date Recorded"] >= start_date)
        & (df["Date Recorded"] <= end_date)
    ].index
    logging.warning(
        f"Dropping {idx.shape[0]} entries with {column} between {start_date} and {end_date} for ID {id}"
    )
    df.drop(idx, inplace=True)
    return df


def _remove_recording(df, id, column, value):
    """
    Removes a recording from a dataframe
    """
    idx = df[(df.ID == id) & (df[column] == value)].index
    logging.warning(
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

    # WARNING:root:r ID 202: FEV1 % Predicted should be in 0-140%, got 147.24696075623552
    # This is the case for many entries for this ID
    # Values from 2020-07-17 to 2020-09-22 judged erroneous. I deleted them because there were too many consecutive erroneous values to correct them meaningfully​
    # 2 values on 2020-09-25 and 2020-10-02 should get corrected by the smoothing​
    df = _remove_recordings_in_date_range(
        df, "202", "FEV1", datetime.date(2020, 7, 17), datetime.date(2020, 9, 22)
    )
    return df


def load_drug_therapies():
    drug_df = pd.read_csv(
        dh.get_path_to_main()
        + "DataFiles/BR/REDCapData/ProcessedData/brDrugTherapy_20240510.csv",
        usecols=[0, 3, 4, 5, 6],
        dtype={
            "ID": str,
        },
    )
    drug_df["DrugTherapyStartDate"] = pd.to_datetime(
        drug_df["DrugTherapyStartDate"]
    ).dt.date
    drug_df["DrugTherapyStopDate"] = pd.to_datetime(
        drug_df["DrugTherapyStopDate"]
    ).dt.date

    def drug_therapy_type_dict():
        return {
            1: "Ivacaftor",
            2: "Trikafta",
            3: "Orkambi",
            4: "Symkevi",
            999: "Unknown",
        }

    drug_df["DrugTherapyType"] = drug_df["DrugTherapyType"].map(
        drug_therapy_type_dict()
    )

    drug_df = drug_df.sort_values(
        by=["ID", "DrugTherapyStartDate"], ascending=[True, False]
    )

    drug_df = _correct_drug_therapies(drug_df)
    drug_df.groupby("ID").apply(sanity_checks.drug_therapies)

    return drug_df


def add_drug_therapy_to_df(df):
    """
    The input df needs to have an ID column and a Date Recorded column
    """
    drug_df = load_drug_therapies()

    df["DrugTherapyType"] = None
    # More efficient to iterate through drug_df rather than iterate through df
    for i, row in drug_df.iterrows():
        mask_ID = df.ID == row.ID
        if pd.isnull(row.DrugTherapyStopDate):
            mask_date = df["Date Recorded"] >= row.DrugTherapyStartDate
        else:
            # Note: Seried.between in inclusive to the right and to the left
            mask_date = df["Date Recorded"].between(
                row.DrugTherapyStartDate, row.DrugTherapyStopDate
            )
        df.loc[mask_ID & mask_date, "DrugTherapyType"] = row.DrugTherapyType
    return df


def _correct_drug_therapies(drug_df):
    idx = drug_df[
        (drug_df["ID"] == "108")
        & (drug_df.DrugTherapyType == "Symkevi")
        & (drug_df.DrugTherapyStartDate == datetime.date(2020, 4, 1))
    ].index
    logging.warning(
        f"ID 108 - Dropping the Symkevi entry as it's got the same start date as an ongoing Trikafta treatment, but was stopped after 1 month"
    )
    drug_df = drug_df.drop(idx)

    idx = drug_df[
        (drug_df["ID"] == "131")
        & (drug_df.DrugTherapyType == "Symkevi")
        & (drug_df.DrugTherapyStopDate == datetime.date(2020, 12, 28))
    ].index
    logging.warning(
        f"ID 131 - Shifting Symkevi stop date by 2 day to avoid overlap with Trikfta start"
    )
    # Replace Stop Date by 2020-12-26 for this index
    drug_df.loc[idx, "DrugTherapyStopDate"] = datetime.date(2020, 12, 26)

    idx = drug_df[
        (drug_df["ID"] == "234")
        & (drug_df.DrugTherapyType == "Ivacaftor")
        & (drug_df.DrugTherapyStopDate == datetime.date(2022, 8, 1))
    ].index
    logging.warning(
        f"ID 234 - Setting Ivacaftor stop date to 2021-06-30 to not overlap with Trikafta start date"
    )
    drug_df.loc[idx, "DrugTherapyStopDate"] = datetime.date(2021, 6, 30)

    idx = drug_df[
        (drug_df["ID"] == "334")
        & (drug_df.DrugTherapyType == "Symkevi")
        & (drug_df.DrugTherapyStopDate == datetime.date(2020, 12, 1))
    ].index
    logging.warning(
        f"ID 334 - Changing Symkevi stop date by 3 months to avoid overlap with Trikafta start"
    )
    drug_df.loc[idx, "DrugTherapyStopDate"] = datetime.date(2020, 8, 31)

    idx = drug_df[
        (drug_df["ID"] == "334")
        & (drug_df.DrugTherapyType == "Symkevi")
        & (drug_df.DrugTherapyStartDate == datetime.date(2021, 8, 1))
    ].index
    logging.warning(
        f"ID 334 - Patient alternates between Symkevi and Trikafta as can't tolerate full Trikafta dose. Let's say he is on Trikafta, thus removing the Symkevi entry"
    )
    drug_df = drug_df.drop(idx)

    idx = drug_df[
        (drug_df["ID"] == "335")
        & (drug_df.DrugTherapyType == "Symkevi")
        & (drug_df.DrugTherapyStartDate == datetime.date(2020, 5, 26))
    ].index
    logging.warning(
        f"ID 335 - Symkevi stop date has probably the wrong year, putting 2020 instead of 2022"
    )
    drug_df.loc[idx, "DrugTherapyStopDate"] = datetime.date(2020, 10, 15)

    idx = drug_df[
        (drug_df["ID"] == "413")
        & (drug_df.DrugTherapyType == "Trikafta")
        & (drug_df.DrugTherapyStartDate == datetime.date(2021, 7, 5))
    ].index
    logging.warning(
        f"ID 335 - Shift Trikafta stop date to 4 days earlier to avoid overlap with Ivacaftor"
    )
    drug_df.loc[idx, "DrugTherapyStopDate"] = datetime.date(2021, 7, 31)

    drug_df = drug_df.reset_index(drop=True)
    return drug_df


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
    logging.info("*** Building O2Sat, FEV1, FEF2575 dataframe ***")

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

    logging.info(
        f"Built data structure with {df.ID.nunique()} IDs and {len(df)} entries"
    )

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
        lambda x: sanity_checks.ecfev1_prct_predicted(x["ecFEV1 % Predicted"], x.ID),
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
