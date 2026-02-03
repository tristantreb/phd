from pgmpy.inference.ExactInference import VariableElimination

import models.builders as mb


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
