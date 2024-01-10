def calc_healthy_O2_sat(height: int, sex: str):
    """
    Healthy/predicted O2 Saturation = a + b*isMale + c*(Height-avg_height)
    """
    # Results from fit done in breate_healthy_O2_modelling.ipynb
    a = 98.08683135024685
    b = -0.8209519722244375
    c = -0.017936971739351878
    avg_height = 166.54545454545453

    std = 1.0304
    if sex == "Female":
        return {
            "mean": a + c * (height - avg_height),
            "sigma": std,
        }
    elif sex == "Male":
        return {
            "mean": a + b + c * (height - avg_height),
            "sigma": std,
        }
    else:
        raise ValueError("Sex '{sex}' not in 'Female' or 'Male'")


def calc_healthy_O2_sat_df(df):
    """
    Returns input DataFrame with added column Healthy O2 Saturation, given Height and Sex
    """
    df["Healthy O2 Saturation"] = df.apply(
        lambda x: calc_healthy_O2_sat(x.Height, x.Sex)["mean"],
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
