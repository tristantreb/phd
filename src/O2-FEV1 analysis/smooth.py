import numpy as np


def smooth_avg(O2_FEV1, var_list, window=3):
    for var in var_list:
        O2_FEV1["{} smoothed".format(var)] = O2_FEV1[var].rolling(window).sum()/3
    return O2_FEV1

def smooth_max(O2_FEV1, var_list, window=3):
    for var in var_list:
        O2_FEV1["{} smoothed".format(var)] = O2_FEV1[var].rolling(window, center=True).max()
    return O2_FEV1

def smooth_avg_of_two_highest(O2_FEV1, var_list, window=3):
    for var in var_list:
        O2_FEV1["{} smoothed".format(var)] = O2_FEV1[var].rolling(window, center=True).apply(lambda x: np.sort(x)[-2:].mean())
    return O2_FEV1