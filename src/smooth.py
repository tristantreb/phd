def smooth_avg(O2_FEV1, var_list, window=3):
    for var in var_list:
        O2_FEV1["{} smoothed".format(var)] = O2_FEV1[var].rolling(window).sum()/3
        print("Removed {} entries after avg smoothing {}, kept {}".format(O2_FEV1["{} smoothed".format(var)].isna().sum(), var, O2_FEV1.shape[0]))
    return O2_FEV1

def smooth_max(O2_FEV1, var_list, window=3):
    for var in var_list:
        O2_FEV1["{} smoothed".format(var)] = O2_FEV1[var].rolling(window).max()
        O2_FEV1["{} smoothed".format(var)].dropna(inplace=True)
        print("Removed {} entries after max smoothing {}, kept {}".format(O2_FEV1["{} smoothed".format(var)].isna().sum(), var, O2_FEV1.shape[0]))
    return O2_FEV1