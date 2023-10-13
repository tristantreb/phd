import numpy as np
import pandas as pd



def compute_hfev1_ld_factor(O2_FEV1):
    """
    Factor function: Unblocked_FEV1(Healthy_FEV1, Lung_Damage)
    We model Unblocked FEV1 as the 3rd highest FEV1 measurement
    hfev1 = healthy fev1, ld = lung damage
    """
    df = pd.DataFrame(
        columns=["ID", "Unblocked FEV1 (L)", "Lung Damage (%)", "Healthy FEV1 (L)"]
    )

    for id in O2_FEV1.ID.unique():
        # For a given patient id, filter the FEV1 measurements
        mask = O2_FEV1["ID"] == id
        O2_FEV1_patient = O2_FEV1[mask]
        # Find the unblocked FEV1 (L). We assume that, over the 6 months study period, the patient has done some measurements where he was not blocked
        # To avoid taking an outlier up, which is third highest FEV1 measurement
        rmax = O2_FEV1_patient["FEV1"].nlargest(3).iloc[-1]
        # Get the theoretical healthy FEV1 (L)
        healthy_fev1 = O2_FEV1_patient["Predicted FEV1"].iloc[0]
        # Compute the Lung damage (%)
        lung_damage = 100 * (1 - rmax / healthy_fev1)
        # Add the patient id, reversed max FEV1 and healthy FEV1 (L) to the dataframe
        new_row = pd.DataFrame(
            {
                "ID": [id],
                "Unblocked FEV1 (L)": [rmax],
                "Lung Damage (%)": [lung_damage],
                "Healthy FEV1 (L)": [healthy_fev1],
            }
        )
        df = pd.concat([df, new_row])
    return df


# Factor function: Unblocked_O2_Sat(Healthy_O2_Sat, Lung_Damage)
# We model Unblocked O2 Saturation as constantly set to 100%
# ho2 = healthy o2 saturation, ld = lung damage
def compute_ho2_ld_factor(O2_FEV1):
    df = pd.DataFrame(
        columns=[
            "ID",
            "Unblocked O2 Saturation (%)",
            "Lung Damage (%)",
            "Healthy O2 Saturation (%)",
        ]
    )
    for id in O2_FEV1.ID.unique():
        # For a given patient id, filter the O2 Saturation measurements
        mask = O2_FEV1["ID"] == id
        O2_FEV1_patient = O2_FEV1[mask]
        # Find the unblocked O2 Saturation (%). We assume that, over the 6 months study period, the patient has done some measurements where he was not blocked
        # To avoid taking an outlier up, which is third highest O2 Saturation measurement
        rmax = O2_FEV1_patient["O2 Saturation"].nlargest(3).iloc[-1]
        # Theoretical O2 saturation is 100%
        healthy_o2_sat = 100
        # Compute the Lung damage (%)
        lung_damage = 100 - rmax  # 100 * (1-rmax/healthy_o2_sat)
        # Add the patient id, reversed max O2 Saturation and healthy O2 Saturation (%) to the dataframe
        new_row = pd.DataFrame(
            {
                "ID": [id],
                "Unblocked O2 Saturation (%)": [rmax],
                "Lung Damage (%)": [lung_damage],
                "Healthy O2 Saturation (%)": [healthy_o2_sat],
            }
        )
        df = pd.concat([df, new_row])
    return df


def compute_avg_lung_func_stable(df, fev1_col="FEV1 % Predicted"):
    """
    1- Computes the FEV1 % Predicted during stable period (i.e. when Is Exacerbated is False)
    2- Adds this column to the input df
    3- Adds the avg FEV1 % Pred in stable period next to the ID to add this information to the plot
    4- Order by avg FEV1 % Pred in stable period

    Generalised to other measures of lung function (e.g. FEV1 L)
    """
    if f"Avg {fev1_col} in stable period" in df.columns:
        return df

    # Compute avg of FEV1 % Predicted during stable period (i.e. when Is Exacerbated is False)
    s_avg_fev1_pred_stable = (
        df[df["Is Exacerbated"] == False]
        .groupby(["ID"])[fev1_col]
        .agg("mean")
        .rename(f"Avg {fev1_col} in stable period")
    )
    # Note: Transform compute an agglomerates but returns a df of the same size. That means the agg values is repeated for each row of the group

    # Merge s_avg_fev1_pred_stable with df based on ID
    df = pd.merge(df, s_avg_fev1_pred_stable, on="ID", how="left").sort_values(
        by=[f"Avg {fev1_col} in stable period"]
    )

    unit = "%" if fev1_col == "FEV1 % Predicted" else "L"
    # Add a new column with the ID and the avg FEV1 % Pred in stable period
    df[f"ID (avg {fev1_col} in stable period)"] = df.apply(
        lambda x: f"{x.ID} ({str(round(x[f'Avg {fev1_col} in stable period'],1))}{unit})",
        axis=1,
    )
    return df


def get_avg_diff_of_means(df, var, ex_col):
    """
    1 - Compute mean of O2 Saturation in Ex and Stable periods based on ex_col.
    2 - Compute mean difference between the two.
    3 - Return the average difference of the means.
    """
    avg_diff_of_means = np.array([])
    for id in df.ID.unique():
        df_for_ID = df[df.ID == id].copy().reset_index(drop=True)

        # If it has more than 30 datapoints
        if df_for_ID.shape[0] < 30:
            continue
        mean_ex = df_for_ID[df_for_ID[ex_col] == 1][var].mean()
        mean_stable = df_for_ID[df_for_ID[ex_col] == 0][var].mean()
        avg_diff_of_means = np.append(avg_diff_of_means, mean_stable - mean_ex)
    return np.nanmean(avg_diff_of_means)
