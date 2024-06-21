import src.o2_fev1_analysis.smooth as smooth


def _effort_corrected_smoothing(df, col, scale, shift):
    """
    Works on df with NaN in col.
    """

    df[f"ec{col}"] = df[col]
    no_nan_mask = ~df[f"ec{col}"].isna()
    df.iloc[no_nan_mask] = smooth.identify_and_replace_outliers_up(
        df[no_nan_mask], f"ec{col}", scale=scale, shift=shift
    ).copy()
    df[f"ec{col}"][no_nan_mask] = smooth.smooth_vector(
        df[f"ec{col}"][no_nan_mask].to_numpy(), "max"
    )

    return df


def calc_smoothed_fe_measures(df):
    """
    Calculates Effort Corrected FEV1, FEF2575 and PEF (L/s), if applicable, for the input DataFrame.
    Works despite NaN values in the columns.
    """

    if "FEV1" in df.columns:
        df = (
            df.groupby(by="ID")
            .apply(lambda x: _effort_corrected_smoothing(x, "FEV1", 3, 0.5))
            .reset_index(drop=True)
        )
    if "FEF2575" in df.columns:
        df = (
            df.groupby(by="ID")
            .apply(lambda x: _effort_corrected_smoothing(x, "FEF2575", 3, 0.5))
            .reset_index(drop=True)
        )
    if "PEF (L/s)" in df.columns:
        df = (
            df.groupby(by="ID")
            .apply(lambda x: _effort_corrected_smoothing(x, "PEF (L/s)", 3, 1))
            .reset_index(drop=True)
        )
    return df


# Old version
# def calc_with_smoothed_max_df(df):
#     """
#     Returns input df with Effort Corrected FEV1
#     """
#     # Create a copy of the DataFrame to avoid SettingWithCopyWarning
#     df = df.sort_values(by=["ID", "Date Recorded"]).copy()

#     # Initialize the new column with NaN values
#     df["ecFEV1"] = np.nan

#     # For each unique ID in df
#     for id in df.ID.unique():
#         # Create a mask for this ID
#         mask = df.ID == id

#         # Adjust outliers up. If measurement is above 140% of the monthly average, take the average of the 2 neighbouring values

#         # Adjust outliers down
#         df.loc[mask, "ecFEV1"] = smooth.smooth_vector(
#             df.loc[mask, "FEV1"].to_numpy(), "max"
#         )
#         # if "FEF2575" in df.columns:
#         #     df.loc[mask, "ecFEF2575"] = smooth.smooth_vector(
#         #         df.loc[mask, "FEF2575"].to_numpy(), "max"
#         #     )
#         # if "PEF" in df.columns:
#         #     df.loc[mask, "ecPEF"] = smooth.smooth_vector(
#         #         df.loc[mask, "PEF"].to_numpy(), "max"
#         #     )

#     return df
