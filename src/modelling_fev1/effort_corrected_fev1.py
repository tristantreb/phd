import numpy as np

import src.o2_fev1_analysis.smooth as smooth


def calc_with_smoothed_max(df):
    """
    Returns input df with Effort Corrected FEV1
    """
    df.sort_values(by=["ID", "Date Recorded"], inplace=True)

    df["ecFEV1"] = np.nan

    # For each each ID in df
    for id in df.ID.unique():
        # Create mask for this ID
        mask = df.ID == id
        df["ecFEV1"][mask] = smooth.smooth_vector(df.FEV1[mask].to_numpy(), "max")
    return df
