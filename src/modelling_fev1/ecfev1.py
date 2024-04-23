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

        # Adjust outliers up. If measurement is above 140% of the monthly average, take the average of the 2 neighbouring values

        # Adjust outliers down
        df.loc[mask, "ecFEV1"] = smooth.smooth_vector(
            df.loc[mask, "FEV1"].to_numpy(), "max"
        )
        # if "FEF2575" in df.columns:
        #     df.loc[mask, "ecFEF2575"] = smooth.smooth_vector(
        #         df.loc[mask, "FEF2575"].to_numpy(), "max"
        #     )
        # if "PEF" in df.columns:
        #     df.loc[mask, "ecPEF"] = smooth.smooth_vector(
        #         df.loc[mask, "PEF"].to_numpy(), "max"
        #     )

    return df
