import numpy as np
import pandas as pd


# Smooth the requested columns of O2-FEV1 with smooth_func
def smooth(O2_FEV1, var_list, mode="mean"):
    # Sort O2_FEV1 by ID and Date Recorded
    O2_FEV1 = O2_FEV1.sort_values(by=["ID", "Date recorded"])

    for var in var_list:
        O2_FEV1["{} {} smoothed".format(var, mode)] = 0.0

        # For each each ID in O2_FEV1
        for id in O2_FEV1["ID"].unique():
            # Create mask for this ID
            mask = O2_FEV1["ID"] == id
            O2_FEV1["{} {} smoothed".format(var, mode)][mask] = smooth_vector(
                O2_FEV1[var][mask].to_numpy(), mode
            )
    return O2_FEV1


# Applies a moving max/mean of size 3 on the input vector
def smooth_vector(vector, mode):
    # Copy the vector
    smoothed_vector = vector.copy()
    # Get the vector length
    n = len(vector)
    # For i going from 1 to vector length
    for i in range(0, n - 1):
        # If i == 1, set the value to the max between the current value and the next two values
        if i == 1:
            smoothed_vector[i] = eval("np." + mode)(
                [vector[i], vector[i + 1], vector[i + 2]]
            )
        # If i == vector length, set the value to the max/mean between the current value and the previous two values
        elif i == n - 1:
            smoothed_vector[i] = eval("np." + mode)(
                [vector[i], vector[i - 1], vector[i - 2]]
            )
        # Else, set the value to the max/mean between the current value and the previous and next values
        else:
            smoothed_vector[i] = eval("np." + mode)(
                [vector[i], vector[i - 1], vector[i + 1]]
            )
    return smoothed_vector
