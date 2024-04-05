from datetime import timedelta

import numpy as np
import pandas as pd


def smooth(O2_FEV1, var_list, mode="max"):
    """
    Smooth the requested columns of O2-FEV1 with smooth_func
    """
    # Sort O2_FEV1 by ID and Date Recorded
    O2_FEV1 = O2_FEV1.sort_values(by=["ID", "Date Recorded"])

    for var in var_list:
        O2_FEV1["{} smoothed".format(var)] = 0.0

        # For each each ID in O2_FEV1
        for id in O2_FEV1["ID"].unique():
            # Create mask for this ID
            mask = O2_FEV1["ID"] == id
            O2_FEV1["{} smoothed".format(var)][mask] = smooth_vector(
                O2_FEV1[var][mask].to_numpy(), mode
            )
    return O2_FEV1


def smooth_vector(vector, mode):
    """
    Applies a moving max/mean of size 3 on the input vector
    """
    # Copy the vector
    smoothed_vector = vector.copy()
    # Get the vector length
    n = len(vector)
    if n == 1:
        return vector[0]
    if n == 2:
        return eval("np." + mode)([vector[0], vector[1]])

    for i in range(0, n - 1):
        smoothed_vector[i] = apply_three_day_window(vector, i, n, mode)

    return smoothed_vector


def apply_three_day_window(vector, i, n, mode="max"):
    # If i == 0, set the value to the max between the current value and the next two values
    if i == 0:
        return eval("np." + mode)([vector[i], vector[i + 1], vector[i + 2]])
    # If i == vector length, set the value to the max/mean between the current value and the previous two values
    elif i == n - 1:
        return eval("np." + mode)([vector[i], vector[i - 1], vector[i - 2]])
    # Else, set the value to the max/mean between the current value and the previous and next values
    else:
        return eval("np." + mode)([vector[i], vector[i - 1], vector[i + 1]])


def replace_with_neighbouring_value(vector, i, n, mode="max"):
    # If i == 0, set the value to the max between the current value and the next two values
    if i == 0:
        if n > 1:
            return eval("np." + mode)([vector[i + 1]])
        return eval("np." + mode)([vector[i]])
    # If i == vector length, set the value to the max/mean between the current value and the previous two values
    elif i == n - 1:
        if n > 1:
            return eval("np." + mode)([vector[i - 1]])
        return eval("np." + mode)([vector[i]])
    # Else, set the value to the max/mean between the current value and the previous and next values
    else:
        return eval("np." + mode)([vector[i - 1], vector[i + 1]])


def identify_and_replace_outliers_up(df, col):
    if len(df) == 1:
        return df
    for i in range(len(df)):
        date = df["Date Recorded"].iloc[i]
        val = df[col].iloc[i]

        date_up = date + timedelta(days=15)
        date_max = max(df["Date Recorded"])
        if date_up >= date_max:
            date_up = date_max

        date_low = date - timedelta(days=15)
        date_min = min(df["Date Recorded"])
        if date_low <= date_min:
            date_low = date_min

        df_in_interval = df[
            (df["Date Recorded"] > date_low) & (df["Date Recorded"] < date_up)
        ]

        mean = df_in_interval[col].mean()

        if val > mean * 1.4:
            new = replace_with_neighbouring_value(df[col].to_numpy(), i, len(df))
            print(
                f"ID {df.ID[0]} - Outlier up for {col}, day {date}: {val} > {mean}, update to {new}"
            )
            df.loc[i, col] = new

    return df
