import modin.pandas as pd
import numpy as np
from pgmpy.inference.ExactInference import VariableElimination

import inf_cutset_conditioning.cutset_cond_algs_learn_ar_change_noo2sat as cca_ar_change_noo2sat
import models.builders as mb


def run_ve(df, with_fef2575=False, with_rmax=False, df_rmax_rows=None):
    """
    Last measurement on row 1
    Robust max FEV1 on row 2
    """
    df = df.reset_index()

    id, height, age, sex = df.iloc[0][["ID", "Height", "Age", "Sex"]]
    # ar_prior = "breathe (2 days model, ecFEV1 addmultnoise, ecFEF25-75)"
    ar_prior = "uniform"
    ecfev1_noise_model_suffix = "_std_add_mult_ecfev1"
    fef2575_cpt_suffix = "_ecfev1_2_days_model_add_mult_noise"

    (
        model,
        HFEV1,
        AR_vars,
        uFEV1_vars,
        ecFEV1_vars,
        ecFEF2575prctecFEV1_vars,
    ) = mb.fev1_fef2575_n_day_BN_noise(
        2 if with_rmax else 1,
        height,
        age,
        sex,
        ar_prior,
        fef2575_cpt_suffix,
        ecfev1_noise_model_suffix,
    )
    var_elim = VariableElimination(model)

    evidence_dict = {}
    if with_fef2575:
        evidence_dict[ecFEF2575prctecFEV1_vars[0].name] = df.loc[
            0, "idx ecFEF2575%ecFEV1"
        ]
    if with_rmax:
        [rmax_fev1] = df_rmax_rows[df_rmax_rows.ID == id]["idx ecFEV1 (L)"].values
        evidence_dict[ecFEV1_vars[1].name] = rmax_fev1
    if with_rmax and with_fef2575:
        [rmax_fef2575] = df_rmax_rows[df_rmax_rows.ID == id][
            "idx ecFEF2575%ecFEV1"
        ].values
        evidence_dict[ecFEF2575prctecFEV1_vars[1].name] = rmax_fef2575

    # print(f"n days: {len(AR_vars)}, evidence_dict: {evidence_dict}")

    res_ve = var_elim.query(
        variables=[ecFEV1_vars[0].name],
        evidence=evidence_dict,
        joint=False,
    )
    dist_ecfev1_ve = res_ve[ecFEV1_vars[0].name].values
    p_ecfev1_ve = dist_ecfev1_ve[df.loc[0, "idx ecFEV1 (L)"]]
    return p_ecfev1_ve


def run_ve_for_ID(
    df,
    colname="log P(FEV1|M1)",
    with_fef2575=False,
    with_rmax=False,
    df_rmax_rows=None,
    drop_cols=True,
):
    """
    Returns log p(FEV1|M), the log probability of the FEV1 data, as new col in df
    Uses exact inference with variable elimination
    Model M selected depending on the two with_x inputs
    Uniform AR prior
    """
    df = df.reset_index(drop=True)

    id, height, age, sex = df.iloc[0][["ID", "Height", "Age", "Sex"]]
    # ar_prior = "breathe (2 days model, ecFEV1 addmultnoise, ecFEF25-75)"
    ar_prior = "uniform"
    ecfev1_noise_model_suffix = "_std_add_mult_ecfev1"
    fef2575_cpt_suffix = "_ecfev1_2_days_model_add_mult_noise"

    (
        model,
        _,
        _,
        _,
        ecFEV1_vars,
        ecFEF2575prctecFEV1_vars,
    ) = mb.fev1_fef2575_n_day_BN_noise(
        2 if with_rmax else 1,
        height,
        age,
        sex,
        ar_prior,
        fef2575_cpt_suffix,
        ecfev1_noise_model_suffix,
    )
    var_elim = VariableElimination(model)

    df[colname] = df.apply(
        lambda row: run_ve_for_ID_entry(
            row,
            var_elim,
            ecFEV1_vars,
            ecFEF2575prctecFEV1_vars,
            with_fef2575,
            with_rmax,
            df_rmax_rows,
        ),
        axis=1,
    )

    if drop_cols:
        cols2keep = ["ID", "Date Recorded", colname]
        return df[cols2keep]

    return df


def run_ve_for_ID_entry(
    row,
    inf,
    ecFEV1_vars,
    ecFEF2575prctecFEV1_vars,
    with_fef2575=False,
    with_rmax=False,
    df_rmax_rows=None,
):
    """
    Returns the log prob of the FEV1 observation for this row
    inf contains the single or two days model depending on the with_rmax value
    """

    evidence_dict = {}
    if with_fef2575:
        evidence_dict[ecFEF2575prctecFEV1_vars[0].name] = row["idx ecFEF2575%ecFEV1"]
    if with_rmax:
        [rmax_fev1] = df_rmax_rows[df_rmax_rows.ID == id]["idx ecFEV1 (L)"].values
        evidence_dict[ecFEV1_vars[1].name] = rmax_fev1
    if with_rmax and with_fef2575:
        [rmax_fef2575] = df_rmax_rows[df_rmax_rows.ID == id][
            "idx ecFEF2575%ecFEV1"
        ].values
        evidence_dict[ecFEF2575prctecFEV1_vars[1].name] = rmax_fef2575

    res_ve = inf.query(
        variables=[ecFEV1_vars[0].name],
        evidence=evidence_dict,
        joint=False,
    )
    dist_ecfev1_ve = res_ve[ecFEV1_vars[0].name].values
    p_ecfev1_ve = dist_ecfev1_ve[row["idx ecFEV1 (L)"]]
    return np.log(p_ecfev1_ve)


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
        get_p_d_given_s=True,
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


def process_data_no_interconnections(df):
    """
    Running this for one day with FEV1 and FEF gives same result than
    the process_data_AR_through_time function
    """
    ar_prior = "breathe (2 days model, ecFEV1 addmultnoise, ecFEF25-75)"
    ecfev1_noise_model_suffix = "_std_add_mult_ecfev1"
    fef2575_cpt_suffix = "_ecfev1_2_days_model_add_mult_noise"

    log_p_S_given_D, _ = (
        cca_ar_change_noo2sat.run_long_noise_model_no_ar_interconnections(
            df,
            ar_prior=ar_prior,
            ecfev1_noise_model_suffix=ecfev1_noise_model_suffix,
            fef2575_cpt_suffix=fef2575_cpt_suffix,
            debug=False,
        )
    )
    return log_p_S_given_D


def process_id_2day_fev1_fef_model(id, df20, df_rmax_rows):
    """
    To get results for the two days model, I run the process_id_longitudinal_data n times,
    each time adding the rmax FEV1 and rmax FEF25-75 as a second day.
    """
    df_for_ID = df20[df20.ID == id]
    df_rmax_row = df_rmax_rows[df_rmax_rows.ID == id]

    log_p_S_given_D = []
    for i, _ in df_for_ID.iterrows():
        dftmp = df_for_ID.iloc[i : i + 1]
        dftmp = pd.concat([df_rmax_row, dftmp]).reset_index()
        log_p_S_given_Di = process_data_no_interconnections(dftmp)
        log_p_S_given_D.append(log_p_S_given_Di)
    return np.sum(log_p_S_given_D)
