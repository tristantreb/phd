import numpy as np


# Applies a moving average of size 3, smooth depending on the smoothing type
def smooth(O2_FEV1, var_to_smooth_list, type="max", window=3):

    # For each var_to_smooth in var_to_smooth_list, create a column with name "{} smoothed".format(var_to_smooth) filled with nan
    for var_to_smooth in var_to_smooth_list:
        O2_FEV1["{} smoothed".format(var_to_smooth)] = np.nan

    # For each each ID in O2_FEV1
    for id in O2_FEV1["ID"].unique():
        # Create mask for this ID
        mask = O2_FEV1["ID"] == id
        for var_to_smooth in var_to_smooth_list:
            if type == "max":
                O2_FEV1["{} smoothed".format(var_to_smooth)][mask] = (
                    O2_FEV1[var_to_smooth][mask].rolling(window, center=True).max()
                )
            elif type == "avg":
                O2_FEV1["{} smoothed".format(var_to_smooth)][mask] = (
                    O2_FEV1[var_to_smooth][mask].rolling(window, center=True).mean()
                )
    return O2_FEV1
