import numpy as np
import pandas as pd

import inf_cutset_conditioning.cutset_cond_algs_learn_ar_change_noo2sat as cca_ar_change_noo2sat


def process_data_AR_through_time(df):
    ecfef2575_cols = [
        "ecFEF2575%ecFEV1",
        "idx ecFEF2575%ecFEV1",
        "idx ecFEF25-75 % ecFEV1 (%)",
    ]
    ecfev1_cols = [
        "ecFEV1",
        "idx ecFEV1 (L)",
    ]

    # Obs FEV1 and FEF25-75
    #
    # Obs FEV1
    # dftmp[ecfef2575_cols] = np.nan
    # Obs no data
    # dftmp[ecfev1_cols + ecfef2575_cols] = np.nan

    ar_change_cpt_suffix = "_shape_factor_single_laplace_1.6"
    ar_prior = "breathe (2 days model, ecFEV1 addmultnoise, ecFEF25-75)"
    n_missing_days_allowed = 1
    ecfev1_noise_model_suffix = "_std_add_mult_ecfev1"
    fef2575_cpt_suffix = "_ecfev1_2_days_model_add_mult_noise"

    out = cca_ar_change_noo2sat.run_long_noise_model_through_time(
        # ) = cca_ar_change.run_long_noise_model_through_time(
        # ) = cca.run_long_noise_model_through_time_light(
        df,
        ar_prior=ar_prior,
        ar_change_cpt_suffix=ar_change_cpt_suffix,
        ecfev1_noise_model_suffix=ecfev1_noise_model_suffix,
        fef2575_cpt_suffix=fef2575_cpt_suffix,
        n_days_consec=n_missing_days_allowed + 1,
        get_p_s_given_d=True,
        debug=False,
    )

    (
        [log_p_S_given_D],
        _,
    ) = out
    return log_p_S_given_D


def process_id_long_model(id, df):
    dftmp = df[df.ID == id]
    return process_data_AR_through_time(dftmp)


def process_id_fev1_fef_model(id, df20):
    """
    To get results for the two days model, I run the process_id_longitudinal_data n times,
    each time adding the rmax FEV1 and rmax FEF25-75 as a second day.
    """
    df_for_ID = df20[df20.ID == id]
    log_p_S_given_D = []
    for i, _ in df_for_ID.iterrows():
        dftmp = df_for_ID.iloc[i : i + 1].reset_index()
        log_p_S_given_Di = process_data_AR_through_time(dftmp)
        log_p_S_given_D.append(log_p_S_given_Di)
    return np.sum(log_p_S_given_D)


def process_id_2day_fev1_fef_model(id, df20, df_rmax_rows):
    """
    To get results for the two days model, I run the process_id_longitudinal_data n times,
    each time adding the rmax FEV1 and rmax FEF25-75 as a second day.
    """
    df_for_ID = df20[df20.ID == id]
    df_row = df_rmax_rows[df_rmax_rows.ID == id]

    log_p_S_given_D = []
    for i, _ in df_for_ID.iterrows():
        dftmp = df_for_ID.iloc[i : i + 1]
        dftmp = pd.concat([dftmp, df_row]).reset_index()
        log_p_S_given_Di = process_data_AR_through_time(dftmp)
        log_p_S_given_D.append(log_p_S_given_Di)
    return np.sum(log_p_S_given_D)

