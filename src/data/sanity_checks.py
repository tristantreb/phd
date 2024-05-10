import logging

import numpy as np


def weight(value, id):
    if value <= 30 or value >= 122:
        logging.warning(
            "ID {} has Weight ({}) outside 30-122 kg range".format(id, value)
        )
    return -1


def pulse(value, id):
    """
    Pusle (BPM) should be in range 40-200 beats per minute
    """
    if value <= 40 or value >= 200:
        logging.warning(
            "ID {} has Pulse ({}) outside 40-200 BPM range".format(id, value)
        )
    return -1


def temperature(value, id):
    """
    Temperature should be in range 35-40 degrees celsius
    """
    if value <= 34 or value >= 40:
        logging.warning(
            "ID {} has Temp ({}) outside 35-40 degC range".format(id, value)
        )
    return -1


def o2_saturation(value, id):
    """
    O2 Saturation should be in range 70-100%
    """
    if value <= 70 or value >= 100:
        logging.warning(
            "ID {} has O2 Saturation ({}) outside 70-100% range".format(id, value)
        )
    return -1


def fev1(value, id):
    if value <= 0.2 or value >= 6:
        logging.warning("ID {} has FEV1 ({}) outside 0.2-6 L range".format(id, value))
    return -1


def fef2575(value, id):
    """
    FEF25-75 should be in 1-10 L/s
    """
    if value <= 0.2 or value >= 9:
        logging.warning(
            "ID {} has FEF25-75 ({}) outside 0.2-9 L/s range".format(id, value)
        )
    return -1


def pef(value, id):
    """
    PEF should be in 1-10 L/s
    """
    if value <= 15 or value >= 980:
        logging.warning(
            "ID {} has FEF25-75 ({}) outside 15-980 L/s range".format(id, value)
        )
    return -1


def sex(value, id):
    """
    Sex should be in Male, Female
    """
    if value not in ("Male", "Female"):
        logging.warning(f"r ID {id}: Sex should be either Male or Female, got {value}")
    return -1


def age(value, id):
    """
    Age should be in 18-70 years
    """
    if value <= 18 or value >= 70:
        logging.warning(f"r ID {id}: Age should be in 18-70 years, got {value}")
    return -1


def height(value, id):
    """
    Height should be in 120-220 cm
    """
    if value <= 120 or value >= 220:
        logging.warning(f"r ID {id}: Height should be in 120-220 cm, got {value}")
    return -1


def predicted_fev1(value, id):
    """
    Predicted FEV1 should be in 2-5.5 L
    """
    if value <= 2 or value >= 5.5:
        logging.warning(f"r ID {id}: Predicted FEV1 should be in 2-5.5 L, got {value}")
    return -1


def fev1_prct_predicted(value, id):
    """
    FEV1 % Predicted should be in 0-140%
    """
    if value <= 0 or value >= 140:
        logging.warning(f"r ID {id}: FEV1 % Predicted should be in 0-140%, got {value}")


def data_types(df):
    for col in df.columns:
        match col:
            case "ID" | "Sex" | "Date Recorded":
                if df[col].dtype != np.dtype("O"):
                    logging.warning(
                        "Expected {col} to be of type object, got {df[col].dtype}"
                    )
            case (
                "Height"
                | "FEV1"
                | "FEF2575"
                | "ecFEV1"
                | "Predicted FEV1"
                | "FEV1 % Predicted"
                | "ecFEV1 % Predicted"
                | "Healthy O2 Saturation"
                | "O2 Saturation % Healthy"
            ):
                if df[col].dtype != np.dtype("float64"):
                    logging.warning(
                        f"Expected {col} to be of type float64, got {df[col].dtype}"
                    )
            case "Age":
                if df[col].dtype != np.dtype("int64"):
                    logging.warning(
                        f"Expected {col} to be of type int64, got {df[col].dtype}"
                    )
            case "DateTime Recorded" | "DOB":
                if df[col].dtype != np.dtype("datetime64[ns]"):
                    logging.warning(
                        f"Expected {col} to be of type datetime64[ns], got {df[col].dtype}"
                    )
            case "O2 Saturation" | "PEF":
                # TODO: Choose int or float
                if df[col].dtype != np.dtype("int64") and df[col].dtype != np.dtype(
                    "float64"
                ):
                    logging.warning(
                        f"Expected {col} to be of type int or float, got {df[col].dtype}"
                    )
            case "PartitionKey" | "StudyNumber":
                continue
            case _:
                raise ValueError(f"Unexpected column {col} in dataframe")


def same_day_measurements(df, id_col_name="ID"):
    logging.info(f"* Checking for same day measurements for {id_col_name} *")

    def check(df):
        if len(df) >= 1:
            logging.warning(
                f"{len(df)} measurements recorded on {df['Date Recorded'].values[0]} for ID {df[id_col_name].values[0]}"
            )
        return -1

    df.groupby([id_col_name, "Date Recorded"]).apply(check)

    return -1


def must_not_have_nan(df):
    """
    Checks for NaN values in dataframe
    """
    if df.isna().sum().sum() >= 1:
        logging.warning(f"{df.isna().sum().sum()} NaN values in dataframe")
    else:
        logging.info("No NaN values found in dataframe")
    return -1
