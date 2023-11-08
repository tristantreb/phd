import numpy as np


def weight(value, id):
    if value < 30 or value > 122:
        print(
            "Warning - ID {} has Weight ({}) outside 30-122 kg range".format(id, value)
        )
    return -1


def pulse(value, id):
    """
    Pusle (BPM) should be in range 40-200 beats per minute
    """
    if value < 40 or value > 200:
        print(
            "Warning - ID {} has Pulse ({}) outside 40-200 BPM range".format(id, value)
        )
    return -1


def temperature(value, id):
    """
    Temperature should be in range 35-40 degrees celsius
    """
    if value < 34 or value > 40:
        print(
            "Warning - ID {} has Temp ({}) outside 35-40 degC range".format(id, value)
        )
    return -1


def o2_saturation(value, id):
    """
    O2 Saturation should be in range 70-100%
    """
    if value < 70 or value > 100:
        print(
            "Warning - ID {} has O2 Saturation ({}) outside 70-100% range".format(
                id, value
            )
        )
    return -1


def fev1(value, id):
    if value < 0.1 or value > 5.5:
        print("Warning - ID {} has FEV1 ({}) outside 0.1-5.5 L range".format(id, value))
    return -1


def sex(value, id):
    """
    Sex should be in Male, Female
    """
    if value not in ("Male", "Female"):
        print(f"Warning for ID {id}: Sex should be either Male or Female, got {value}")
    return -1


def age(value, id):
    """
    Age should be in 18-70 years
    """
    if value < 18 or value > 70:
        print(f"Warning for ID {id}: Age should be in 18-70 years, got {value}")
    return -1

def height(value, id):
    """
    Height should be in 120-220 cm
    """
    if value < 120 or value > 220:
        print(f"Warning for ID {id}: Height should be in 120-220 cm, got {value}")
    return -1

def predicted_fev1(value, id):
    """
    Predicted FEV1 should be in 2-5.5 L
    """
    if value < 2 or value > 5.5:
        print(f"Warning for ID {id}: Predicted FEV1 should be in 2-5.5 L, got {value}")
    return -1


def data_types(df):
    for col in df.columns:
        match col:
            case "ID" | "Sex" | "Date Recorded":
                if df[col].dtype != np.dtype("O"):
                  print("Warning - Expected {col} to be of type object, got {df[col].dtype}"")
            case "Height" | "FEV1":
                if df[col].dtype != np.dtype("float64"):
                  print(f"Warning - Expected {col} to be of type float64, got {df[col].dtype}")
            case "Age":
                if df[col].dtype != np.dtype("int64"):
                  print(f"Warning - Expected {col} to be of type int64, got {df[col].dtype}")
            case "DateTime Recorded":
                if df[col].dtype != np.dtype("datetime64[ns]"):
                  print(f"Warning - Expected {col} to be of type datetime64[ns], got {df[col].dtype}")
            case "O2 Saturation":
                # TODO: Choose int or float
                if df[col].dtype != np.dtype("int64") and df[col].dtype != np.dtype("float64"):
                  print(f"Warning - Expected {col} to be of type int or float, got {df[col].dtype}")
            case _:
                raise ValueError(f"Unexpected column {col} in dataframe")
