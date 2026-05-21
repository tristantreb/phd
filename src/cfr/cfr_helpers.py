import numpy as np
from pgmpy.inference.ExactInference import VariableElimination
import pandas as pd

import models.builders as mb


def compute_hfev1_priors(df, bFEV1="best FEV1"):
    # df["P(HFEV1|FEF2575, bFEV1, FEV1)"] = df.apply(infer_hfev1_pers, axis=1)
    df["P(HFEV1|FEV1)"] = df.apply(infer_hfev1_truncation, axis=1)
    df["P(HFEV1|bFEV1)"] = df.apply(
        lambda row: infer_hfev1_truncation(row, fev1_col=bFEV1), axis=1
    )
    return df


def run_ve(row):
    """
    Last measurement on row 1
    Robust max FEV1 on row 2
    """
    id, height, age, sex = row[["ID", "Height", "Age", "Sex"]]
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
        2,
        height,
        age,
        sex,
        ar_prior,
        fef2575_cpt_suffix,
        ecfev1_noise_model_suffix,
    )
    var_elim = VariableElimination(model)

    evidence_dict = {}
    evidence_dict[ecFEV1_vars[0].name] = row["idx FEV1"]
    evidence_dict[ecFEV1_vars[1].name] = row["idx best FEV1"]
    evidence_dict[ecFEF2575prctecFEV1_vars[0].name] = row["idx FEF2575%FEV1"]

    res_ve = var_elim.query(
        variables=[AR_vars[0].name],
        evidence=evidence_dict,
        joint=False,
    )
    dist_ar = res_ve[AR_vars[0].name].values
    return dist_ar


def infer_hfev1_truncation(row, fev1_col="FEV1"):
    """
    FEV1 for soft truncation
    best FEV1 for full truncation
    """
    id, height, age, sex = row[["ID", "Height", "Age", "Sex"]]
    # ar_prior = "breathe (2 days model, ecFEV1 addmultnoise, ecFEF25-75)"
    ar_prior = "uniform"
    ecfev1_noise_model_suffix = "_std_add_mult_ecfev1"
    fef2575_cpt_suffix = "_ecfev1_2_days_model_add_mult_noise"

    (
        model,
        HFEV1,
        _,
        _,
        ecFEV1,
        _,
    ) = mb.fev1_fef2575_1_day_BN_noise(
        height,
        age,
        sex,
        ar_prior,
        fef2575_cpt_suffix,
        ecfev1_noise_model_suffix,
    )

    var_elim = VariableElimination(model)

    evidence_dict = {}

    evidence_dict[ecFEV1.name] = row[f"idx {fev1_col}"]

    res_ve = var_elim.query(
        variables=[HFEV1.name],
        evidence=evidence_dict,
        joint=False,
    )
    dist_hfev1_trunc = res_ve[HFEV1.name].values
    return dist_hfev1_trunc


def infer_hfev1_pers(row, with_fef2575=True, bFEV1="best FEV1"):
    """
    Personalised HFEV1 inference
    """
    id, height, age, sex = row[["ID", "Height", "Age", "Sex"]]
    # ar_prior = "breathe (2 days model, ecFEV1 addmultnoise, ecFEF25-75)"
    ar_prior = "uniform"
    ecfev1_noise_model_suffix = "_std_add_mult_ecfev1"
    fef2575_cpt_suffix = "_ecfev1_2_days_model_add_mult_noise"

    (
        model,
        HFEV1,
        _,
        _,
        ecFEV1_vars,
        ecFEF2575prctecFEV1_vars,
    ) = mb.fev1_fef2575_n_day_BN_noise(
        2,
        height,
        age,
        sex,
        ar_prior,
        fef2575_cpt_suffix,
        ecfev1_noise_model_suffix,
    )
    var_elim = VariableElimination(model)

    evidence_dict = {}
    evidence_dict[ecFEV1_vars[0].name] = row["idx FEV1"]
    evidence_dict[ecFEV1_vars[1].name] = row[f"idx {bFEV1}"]
    if with_fef2575:
        evidence_dict[ecFEF2575prctecFEV1_vars[0].name] = row["idx FEF2575%FEV1"]

    res_ve = var_elim.query(
        variables=[HFEV1.name],
        evidence=evidence_dict,
        joint=False,
    )
    dist_hfev1_pers = res_ve[HFEV1.name].values
    return dist_hfev1_pers


def infer_fev1_pred(row):
    id, height, age, sex = row[["ID", "Height", "Age", "Sex"]]
    # ar_prior = "breathe (2 days model, ecFEV1 addmultnoise, ecFEF25-75)"
    ar_prior = "uniform"
    ecfev1_noise_model_suffix = "_std_add_mult_ecfev1"
    fef2575_cpt_suffix = "_ecfev1_2_days_model_add_mult_noise"

    # Use previously computed hfev1_prior
    hfev1_prior = {"type": "custom", "p": row["P(HFEV1|FEF2575, bFEV1)"]}

    (
        model,
        HFEV1,
        AR,
        uFEV1,
        ecFEV1,
        ecFEF2575prctecFEV1,
    ) = mb.fev1_fef2575_1_day_BN_noise(
        height,
        age,
        sex,
        ar_prior,
        fef2575_cpt_suffix,
        ecfev1_noise_model_suffix,
        hfev1_prior=hfev1_prior,
    )

    # Ensure that HFEV1.cpt is set to previously computed values, as a safeguard.
    # assert np.allclose(HFEV1.cpt, row["P(HFEV1|FEF2575, bFEV1)"])

    var_elim = VariableElimination(model)

    evidence_dict = {}
    # Observe airway resistance to 0%
    evidence_dict[AR.name] = 0

    res_ve = var_elim.query(
        variables=[ecFEV1.name],
        evidence=evidence_dict,
        joint=False,
    )
    dist_ecfev1 = res_ve[ecFEV1.name].values
    return dist_ecfev1


def get_p_data_under_model(row):
    """ """
    id, height, age, sex = row[["ID", "Height", "Age", "Sex"]]
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
        2,
        height,
        age,
        sex,
        ar_prior,
        fef2575_cpt_suffix,
        ecfev1_noise_model_suffix,
    )
    var_elim = VariableElimination(model)

    # P(FEV1|M)
    evidence_dict = {}
    res_ve = var_elim.query(
        variables=[ecFEV1_vars[0].name],
        evidence=evidence_dict,
        joint=False,
    )
    dist_fev1 = res_ve[ecFEV1_vars[0].name].values
    p_fev1 = dist_fev1[row["idx FEV1"]]

    # P(FEF25-75 % FEV1 | M, FEV1)
    evidence_dict = {}
    evidence_dict[ecFEV1_vars[0].name] = row["idx FEV1"]

    res_ve = var_elim.query(
        variables=[ecFEF2575prctecFEV1_vars[0].name],
        evidence=evidence_dict,
        joint=False,
    )
    dist_fefprctfev1 = res_ve[ecFEF2575prctecFEV1_vars[0].name].values
    p_fefprctfev1 = dist_fefprctfev1[row["idx FEF2575%FEV1"]]

    # P(best FEV1 | FEF2575%FEV1, FEV1, M)
    evidence_dict = {}
    evidence_dict[ecFEV1_vars[0].name] = row["idx FEV1"]
    evidence_dict[ecFEF2575prctecFEV1_vars[0].name] = row["idx FEF2575%FEV1"]
    res_ve = var_elim.query(
        variables=[ecFEV1_vars[1].name],
        evidence=evidence_dict,
        joint=False,
    )
    dist_bfev1 = res_ve[ecFEV1_vars[1].name].values
    p_bfev1 = dist_bfev1[row["idx best FEV1"]]

    p_d_given_m = p_fev1 * p_fefprctfev1 * p_bfev1

    return pd.Series([p_fev1, p_fefprctfev1, p_bfev1, p_d_given_m])
