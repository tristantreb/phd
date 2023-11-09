def calc_healthy_O2_sat(sex: str, height: int):
    """
    Healthy/predicted O2 Saturation = a + b*isMale + c*(Height-avg_height)
    """
    # Results from fit done in breate_healthy_O2_modelling.ipynb
    a = 98.04948738170349
    b = -0.5937158858259677
    c = -0.008367002391796437
    avg_height = 166

    std = 1.0877
    if sex == "Female":
        return {
            "mean": a + c * (height - avg_height),
            "std": std,
        }
    elif sex == "Male":
        return {
            "mean": a + b + c * (height - avg_height),
            "std": std,
        }
    else:
        raise ValueError("Sex '{sex}' not in 'Female' or 'Male'")


def calc_healthy_O2_sat_df(df):
    """
    Returns input DataFrame with added column Healthy O2 Saturation, given Height and Sex
    """
    df["Healthy O2 Saturation"] = df.apply(
        lambda x: calc_healthy_O2_sat(x.Sex, x.Height)["mean"],
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
