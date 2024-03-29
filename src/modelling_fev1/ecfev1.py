import numpy as np

import src.o2_fev1_analysis.smooth as smooth


def calc_with_smoothed_max_df(df):
    """
    Returns input df with Effort Corrected FEV1
    """
    # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    df = df.sort_values(by=["ID", "Date Recorded"]).copy()

    # Initialize the new column with NaN values
    df["ecFEV1"] = np.nan

    # For each unique ID in df
    for id in df.ID.unique():
        # Create a mask for this ID
        mask = df.ID == id
        # Use loc to avoid SettingWithCopyWarning and assign the smoothed values
        df.loc[mask, "ecFEV1"] = smooth.smooth_vector(df.loc[mask, "FEV1"].to_numpy(), "max")

    return df