import numpy as np


def smooth_avg(O2_FEV1, var_list, window=3):
    for var in var_list:
        O2_FEV1["{} smoothed".format(var)] = O2_FEV1[var].rolling(window).sum() / 3
    return O2_FEV1


# Applies a moving average of size 3, then take the max of the 3 values
def smooth_max(O2_FEV1, var_to_smooth_list, window=3):

    # For each var_to_smooth in var_to_smooth_list, create a column with name "{} smoothed".format(var_to_smooth) filled with nan
    for var_to_smooth in var_to_smooth_list:
        O2_FEV1["{} smoothed".format(var_to_smooth)] = np.nan

    # For each each ID in O2_FEV1
    for id in O2_FEV1["ID"].unique():
        # Create mask for this ID
        mask = O2_FEV1["ID"] == id
        for var_to_smooth in var_to_smooth_list:
            O2_FEV1["{} smoothed".format(var_to_smooth)][mask] = (
                O2_FEV1[var_to_smooth][mask].rolling(window, center=True).max()
            )
    return O2_FEV1
