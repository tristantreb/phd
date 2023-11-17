def calc_healthy_O2_sat(sex: str, height: int):
    """
    Healthy/predicted O2 Saturation = a + b*isMale + c*(Height-avg_height)
    """
    # Results from fit done in breate_healthy_O2_modelling.ipynb
    a = 98.06752062527138
    b = -0.7384058258246333
    c = -0.012547943971150761
    avg_height = 164.77914110429447

    std = 1.0304
    if sex == "Female":
        return {
            "M": a + c * (height - avg_height),
            "std": std,
        }
    elif sex == "Male":
        return {
            "M": a + b + c * (height - avg_height),
            "std": std,
        }
    else:
        raise ValueError("Sex '{sex}' not in 'Female' or 'Male'")


def calc_healthy_O2_sat_df(df):
    """
    Returns input DataFrame with added column Healthy O2 Saturation, given Height and Sex
    """
    df["Healthy O2 Saturation"] = df.apply(
        lambda x: calc_healthy_O2_sat(x.Sex, x.Height)["M"],
        axis=1,
    )

    return df


def calc_O2_sat_prct_healthy_df(df):
    """
    Returns input DataFramce with added column O2 Saturation % Healthy, given O2 Saturation and Healthy O2 Saturation
    """
    df["O2 Saturation % Healthy"] = (
        df["O2 Saturation"] / df["Healthy O2 Saturation"] * 100
    )
    return df
