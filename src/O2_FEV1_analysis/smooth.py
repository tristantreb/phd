import logging
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
        smoothed_vector[i] = apply_three_day_window(vector, i, mode)

    return smoothed_vector


def smooth_vector_conservative(vector, mode, t=0.8):
    """
    Compute a moving max/mean of size 3 on the input vector
    If the computed value is < 80% away from the original value, take the original value
    """
    # Copy the vector
    smoothed_vector = vector.copy()
    # Get the vector length
    n = len(vector)
    if n == 1:
        return vector[0]
    if n == 2:
        # return eval("np." + mode)([vector[0], vector[1]])
        max_idx = np.argmax([vector[0], vector[1]])
        min_idx = np.argmin([vector[0], vector[1]])
        # If value 1 is t% away from value 2, take value 1
        if vector[min_idx] < t * vector[max_idx]:
            smoothed_vector[min_idx] = vector[max_idx]
    else:
        for i in range(0, n):
            computed_value = apply_three_day_window(vector, i, mode)
            if vector[i] < t * computed_value:
                smoothed_vector[i] = computed_value

    return smoothed_vector


def apply_three_day_window(vector, i, mode="max"):
    """
    Vector is an array of non nan values
    """
    n = len(vector)
    # If i == 0, set the value to the max between the current value and the next two values
    if i == 0:
        return eval("np." + mode)([vector[i], vector[i + 1], vector[i + 2]])
    # If i == vector length, set the value to the max/mean between the current value and the previous two values
    elif i == n - 1:
        return eval("np." + mode)([vector[i], vector[i - 1], vector[i - 2]])
    # Else, set the value to the max/mean between the current value and the previous and next values
    else:
        return eval("np." + mode)([vector[i], vector[i - 1], vector[i + 1]])


def get_previous_val(vector, i, mode="max"):
    """
    Vector is an array of non nan values
    """
    n = len(vector)
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


def identify_and_replace_outliers_up(df, col, scale=1.3, shift=0.5):
    """
    df[col] must not have NaN

    Identify outliers, and replacce them with the previous val if applicable
    """
    df = df.reset_index(drop=True)
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
        # Remove current value
        df_in_interval = pd.concat([df_in_interval[0:i], df_in_interval[i + 1 :]])

        # If there are no other data points in the data range, continue
        if len(df_in_interval) == 0:
            continue

        # Take the 70-95th percentile of the data
        # mean = df_in_interval[col].quantile([0.5, 0.95]).mean()
        mean = df_in_interval[col].mean()
        std = df_in_interval[col].std()

        if (val > mean + scale * std) and (val > mean + shift):
            # Always take the previous value, except if the cur val is the first of the date range
            if i == df_in_interval.index[0]:
                new = df[col].iloc[i + 1]
            else:
                new = df[col].iloc[i - 1]
            logging.info(
                f"ID {df.ID[0]} - Outlier up for {col}, day {date}: {val:.2f} > {mean + scale*std:.2f} and > {mean + shift:.2f} (mean={mean:.2f},std={std:.2f}), update to {new:.2f}"
            )
            df.loc[i, col] = new

    return df
