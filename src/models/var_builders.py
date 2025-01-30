"""
Use functions in this file to define the model's variables, their name, discretisation parameters, and priors (if parent) or conditional probability tables (if child)

Each function corresponds to a full set of variables to be plugged into a bayesian network
"""

import numpy as np

from src.modelling_ar import ar
from src.modelling_ar.ar import (
    get_prior_for_uniform_hfev1_message,
    get_uniform_prior_in_log_space,
)
from src.modelling_o2.ia import get_IA_breathe_prior
from src.models.cpts.helpers import get_cpt
from src.models.helpers import (
    CutsetConditionedTemporalVariableNode,
    DiscreteVariableNode,
    SharedVariableNode,
    TemporalVariableNode,
    VariableNode,
    abbr_to_name,
    get_p_in_log,
)


def fev1_point_in_time(height, age, sex):
    """
    Point in time model with full FEV1 side

    """
    hfev1_prior = {"type": "default", "height": height, "age": age, "sex": sex}
    HFEV1 = VariableNode("Healthy FEV1 (L)", 1, 6, 0.05, prior=hfev1_prior)
    ecFEV1 = VariableNode("ecFEV1 (L)", 0, 6, 0.05, prior=None)
    AR = VariableNode("Airway resistance (%)", 0, 90, 2, prior={"type": "uniform"})
    ecFEV1.set_cpt(get_cpt([ecFEV1, HFEV1, AR]))
    return HFEV1, ecFEV1, AR


def o2sat_fev1_point_in_time(height, age, sex):
    """
    Point in time model with full FEV1 and O2Sat sides

    There is no factor linking AR and IA in this model
    The priors for AR, IA are uniform
    """
    hfev1_prior = {"type": "default", "height": height, "age": age, "sex": sex}
    ho2sat_prior = {
        "type": "default",
        "height": height,
        "sex": sex,
    }

    HFEV1 = VariableNode("Healthy FEV1 (L)", 1, 6, 0.05, prior=hfev1_prior)
    ecFEV1 = VariableNode("ecFEV1 (L)", 0, 6, 0.05, prior=None)
    # Lowest predicted FEV1 is 15% (AR = 1-predictedFEV1)
    AR = VariableNode("Airway resistance (%)", 0, 90, 2, prior={"type": "uniform"})

    # Res 0.5 takes 19s, res 0.2 takes 21s
    HO2Sat = VariableNode("Healthy O2 saturation (%)", 90, 100, 0.5, prior=ho2sat_prior)
    # Highest drop is 92% (for AR = 90%)
    # Hence the lowest O2SatFFA is 90 * 0.92 = 82.8
    O2SatFFA = VariableNode(
        "O2 saturation if fully functional alveoli (%)", 80, 100, 0.5, prior=None
    )
    # O2 sat can't be below 70%.
    # If there's no airway resistance, it should still be possible to reach 70% O2 sat
    # Hence, min IA is 30% because i
    IA = VariableNode("Inactive alveoli (%)", 0, 30, 1, prior={"type": "uniform"})
    # In reality O2 sat can't be below 70%.
    # However, the CPT should account for the fact that the lowest O2 sat is 82.8%.
    # 82.8-30 = 52.8%
    # TODO: should we hardcode the fact that the sum of AR and IA should not be below 70% O2 Sat?
    UO2Sat = VariableNode("Underlying O2 saturation (%)", 50, 100, 0.5, prior=None)
    O2Sat = VariableNode("O2 saturation (%)", 49.5, 100.5, 1, prior=None)

    # Calculate CPTs
    ecFEV1.set_cpt(get_cpt([ecFEV1, HFEV1, AR]))
    O2SatFFA.set_cpt(get_cpt([O2SatFFA, HO2Sat, AR]))
    UO2Sat.set_cpt(get_cpt([UO2Sat, O2SatFFA, IA]))
    O2Sat.set_cpt(get_cpt([O2Sat, UO2Sat]))

    return HFEV1, ecFEV1, AR, HO2Sat, O2SatFFA, IA, UO2Sat, O2Sat


def o2sat_fev1_point_in_time_cf_priors(height, age, sex, ar_prior, ia_prior):
    """
    Point in time model with full FEV1 and O2Sat sides

    There is no factor linking AR and IA in this model
    The priors for AR, IA are learnt from the Breathe data
    """
    hfev1_prior = {"type": "default", "height": height, "age": age, "sex": sex}
    ho2sat_prior = {
        "type": "default",
        "height": height,
        "sex": sex,
    }

    HFEV1 = VariableNode("Healthy FEV1 (L)", 1, 6, 0.05, prior=hfev1_prior)
    ecFEV1 = VariableNode("ecFEV1 (L)", 0, 6, 0.05, prior=None)
    # Lowest predicted FEV1 is 15% (AR = 1-predictedFEV1)
    AR = VariableNode(
        "Airway resistance (%)", 0, 90, 2, prior={"type": "custom", "p": ar_prior}
    )

    # Res 0.5 takes 19s, res 0.2 takes 21s
    HO2Sat = VariableNode("Healthy O2 saturation (%)", 90, 100, 0.5, prior=ho2sat_prior)
    # Highest drop is 92% (for AR = 90%)
    # Hence the lowest O2SatFFA is 90 * 0.92 = 82.8
    O2SatFFA = VariableNode(
        "O2 saturation if fully functional alveoli (%)", 80, 100, 0.5, prior=None
    )
    # O2 sat can't be below 70%.
    # If there's no airway resistance, it should still be possible to reach 70% O2 sat
    # Hence, min IA is 30% because i
    IA = VariableNode(
        "Inactive alveoli (%)", 0, 30, 1, prior={"type": "custom", "p": ia_prior}
    )
    # In reality O2 sat can't be below 70%.
    # However, the CPT should account for the fact that the lowest O2 sat is 82.8%.
    # 82.8-30 = 52.8%
    # TODO: should we hardcode the fact that the sum of AR and IA should not be below 70% O2 Sat?
    UO2Sat = VariableNode("Underlying O2 saturation (%)", 50, 100, 0.5, prior=None)
    O2Sat = VariableNode("O2 saturation (%)", 49.5, 100.5, 1, prior=None)

    # Calculate CPTs
    ecFEV1.set_cpt(get_cpt([ecFEV1, HFEV1, AR]))
    O2SatFFA.set_cpt(get_cpt([O2SatFFA, HO2Sat, AR]))
    UO2Sat.set_cpt(get_cpt([UO2Sat, O2SatFFA, IA]))
    O2Sat.set_cpt(get_cpt([O2Sat, UO2Sat]))

    return HFEV1, ecFEV1, AR, HO2Sat, O2SatFFA, IA, UO2Sat, O2Sat


def o2sat_fev1_point_in_time_cf_ia_prior(height, age, sex):
    """
    Point in time model with full FEV1 and O2Sat sides

    There is no factor linking AR and IA in this model
    The prior for IA is learnt from the Breathe data
    """
    hfev1_prior = {"type": "default", "height": height, "age": age, "sex": sex}
    ho2sat_prior = {
        "type": "default",
        "height": height,
        "sex": sex,
    }

    HFEV1 = VariableNode("Healthy FEV1 (L)", 1, 6, 0.05, prior=hfev1_prior)
    ecFEV1 = VariableNode("ecFEV1 (L)", 0, 6, 0.05, prior=None)
    # Lowest predicted FEV1 is 15% (AR = 1-predictedFEV1)
    AR = VariableNode("Airway resistance (%)", 0, 90, 2, prior={"type": "uniform"})

    # Res 0.5 takes 19s, res 0.2 takes 21s
    HO2Sat = VariableNode("Healthy O2 saturation (%)", 90, 100, 0.5, prior=ho2sat_prior)
    # Highest drop is 92% (for AR = 90%)
    # Hence the lowest O2SatFFA is 90 * 0.92 = 82.8
    O2SatFFA = VariableNode(
        "O2 saturation if fully functional alveoli (%)", 80, 100, 0.5, prior=None
    )
    # O2 sat can't be below 70%.
    # If there's no airway resistance, it should still be possible to reach 70% O2 sat
    # Hence, min IA is 30% because i
    IA = VariableNode(
        "Inactive alveoli (%)",
        0,
        30,
        1,
        prior={"type": "custom", "p": get_IA_breathe_prior()},
    )
    # In reality O2 sat can't be below 70%.
    # However, the CPT should account for the fact that the lowest O2 sat is 82.8%.
    # 82.8-30 = 52.8%
    # TODO: should we hardcode the fact that the sum of AR and IA should not be below 70% O2 Sat?
    UO2Sat = VariableNode("Underlying O2 saturation (%)", 50, 100, 0.5, prior=None)
    O2Sat = VariableNode("O2 saturation (%)", 49.5, 100.5, 1, prior=None)

    # Calculate CPTs
    ecFEV1.set_cpt(get_cpt([ecFEV1, HFEV1, AR]))
    O2SatFFA.set_cpt(get_cpt([O2SatFFA, HO2Sat, AR]))
    UO2Sat.set_cpt(get_cpt([UO2Sat, O2SatFFA, IA]))
    O2Sat.set_cpt(get_cpt([O2Sat, UO2Sat]))

    return HFEV1, ecFEV1, AR, HO2Sat, O2SatFFA, IA, UO2Sat, O2Sat


def o2sat_fev1_point_in_time_model_ar_ia_factor(height, age, sex):
    """
    Point in time model with full FEV1 and O2Sat sides

    AR's prior is uniform
    IA is linked to AR by a factor
    """
    hfev1_prior = {"type": "default", "height": height, "age": age, "sex": sex}
    ho2sat_prior = {
        "type": "default",
        "height": height,
        "sex": sex,
    }

    HFEV1 = VariableNode("Healthy FEV1 (L)", 1, 6, 0.05, prior=hfev1_prior)
    ecFEV1 = VariableNode("ecFEV1 (L)", 0, 6, 0.05, prior=None)
    # Lowest predicted FEV1 is 15% (AR = 1-predictedFEV1)
    AR = VariableNode("Airway resistance (%)", 0, 90, 2, prior={"type": "uniform"})

    # Res 0.5 takes 19s, res 0.2 takes 21s
    HO2Sat = VariableNode("Healthy O2 saturation (%)", 90, 100, 0.5, prior=ho2sat_prior)
    # Highest drop is 92% (for AR = 90%)
    # Hence the lowest O2SatFFA is 90 * 0.92 = 82.8
    O2SatFFA = VariableNode(
        "O2 saturation if fully functional alveoli (%)", 80, 100, 0.5, prior=None
    )
    # O2 sat can't be below 70%.
    # If there's no airway resistance, it should still be possible to reach 70% O2 sat
    # Hence, min IA is 30% because i
    IA = VariableNode("Inactive alveoli (%)", 0, 30, 1, prior=None)
    # In reality O2 sat can't be below 70%.
    # However, the CPT should account for the fact that the lowest O2 sat is 82.8%.
    # 82.8-30 = 52.8%
    # TODO: should we hardcode the fact that the sum of AR and IA should not be below 70% O2 Sat?
    UO2Sat = VariableNode("Underlying O2 saturation (%)", 50, 100, 0.5, prior=None)
    O2Sat = VariableNode("O2 saturation (%)", 49.5, 100.5, 1, prior=None)

    # Calculate CPTs
    ecFEV1.set_cpt(get_cpt([ecFEV1, HFEV1, AR]))
    O2SatFFA.set_cpt(get_cpt([O2SatFFA, HO2Sat, AR]))
    UO2Sat.set_cpt(get_cpt([UO2Sat, O2SatFFA, IA]))
    O2Sat.set_cpt(get_cpt([O2Sat, UO2Sat]))
    IA.set_cpt(get_cpt([IA, AR]))

    return HFEV1, ecFEV1, AR, HO2Sat, O2SatFFA, IA, UO2Sat, O2Sat


def o2sat_fev1_point_in_time_model_ar_ia_factor_test(
    height, age, sex, ar_prior, ar_ia_cpd
):
    """
    Point in time model with full FEV1 and O2Sat sides

    AR's prior is uniform
    IA is linked to AR by a factor
    """
    hfev1_prior = {"type": "default", "height": height, "age": age, "sex": sex}
    ho2sat_prior = {
        "type": "default",
        "height": height,
        "sex": sex,
    }

    HFEV1 = VariableNode("Healthy FEV1 (L)", 1, 6, 0.05, prior=hfev1_prior)
    ecFEV1 = VariableNode("ecFEV1 (L)", 0, 6, 0.05, prior=None)
    # Lowest predicted FEV1 is 15% (AR = 1-predictedFEV1)
    AR = VariableNode(
        "Airway resistance (%)", 0, 90, 2, prior={"type": "custom", "p": ar_prior}
    )

    # Res 0.5 takes 19s, res 0.2 takes 21s
    HO2Sat = VariableNode("Healthy O2 saturation (%)", 90, 100, 0.5, prior=ho2sat_prior)
    # Highest drop is 92% (for AR = 90%)
    # Hence the lowest O2SatFFA is 90 * 0.92 = 82.8
    O2SatFFA = VariableNode(
        "O2 saturation if fully functional alveoli (%)", 80, 100, 0.5, prior=None
    )
    # O2 sat can't be below 70%.
    # If there's no airway resistance, it should still be possible to reach 70% O2 sat
    # Hence, min IA is 30% because i
    IA = VariableNode("Inactive alveoli (%)", 0, 30, 1, prior=None)
    # In reality O2 sat can't be below 70%.
    # However, the CPT should account for the fact that the lowest O2 sat is 82.8%.
    # 82.8-30 = 52.8%
    # TODO: should we hardcode the fact that the sum of AR and IA should not be below 70% O2 Sat?
    UO2Sat = VariableNode("Underlying O2 saturation (%)", 50, 100, 0.5, prior=None)
    O2Sat = VariableNode("O2 saturation (%)", 49.5, 100.5, 1, prior=None)

    # Calculate CPTs
    ecFEV1.set_cpt(get_cpt([ecFEV1, HFEV1, AR]))
    O2SatFFA.set_cpt(get_cpt([O2SatFFA, HO2Sat, AR]))
    UO2Sat.set_cpt(get_cpt([UO2Sat, O2SatFFA, IA]))
    O2Sat.set_cpt(get_cpt([O2Sat, UO2Sat]))
    IA.set_cpt(ar_ia_cpd)

    return HFEV1, ecFEV1, AR, HO2Sat, O2SatFFA, IA, UO2Sat, O2Sat


def o2sat_fev1_point_in_time_model_shared_healthy_vars(
    height, age, sex, ia_prior="uniform", ar_prior="uniform"
):
    """
    Point in time model with full FEV1 and O2Sat sides

    There is no factor linking AR and IA in this model
    The priors for AR, IA are uniform
    """
    hfev1_prior = {"type": "default", "height": height, "age": age, "sex": sex}
    ho2sat_prior = {
        "type": "default",
        "height": height,
        "sex": sex,
    }

    HFEV1 = SharedVariableNode("Healthy FEV1 (L)", 1, 6, 0.05, prior=hfev1_prior)
    ecFEV1 = VariableNode("ecFEV1 (L)", 0, 6, 0.05, prior=None)
    # Lowest predicted FEV1 is 15% (AR = 1-predictedFEV1)
    AR = VariableNode("Airway resistance (%)", 0, 90, 2, prior=None)
    if ar_prior == "uniform":
        AR.cpt = AR.set_prior({"type": "uniform"})
    elif ar_prior == "uniform in log space":
        AR.cpt = AR.set_prior(
            {"type": "custom", "p": get_uniform_prior_in_log_space(AR)}
        )
    elif ar_prior == "uniform message to HFEV1":
        AR.cpt = AR.set_prior(
            {"type": "custom", "p": get_prior_for_uniform_hfev1_message(AR)}
        )

    # Res 0.5 takes 19s, res 0.2 takes 21s
    HO2Sat = SharedVariableNode(
        "Healthy O2 saturation (%)", 90, 100, 0.5, prior=ho2sat_prior
    )
    # Highest drop is 92% (for AR = 90%)
    # Hence the lowest O2SatFFA is 90 * 0.92 = 82.8
    O2SatFFA = VariableNode(
        "O2 saturation if fully functional alveoli (%)", 80, 100, 0.5, prior=None
    )
    # O2 sat can't be below 70%.
    # If there's no airway resistance, it should still be possible to reach 70% O2 sat
    # Hence, min IA is 30% because i
    if ia_prior == "uniform":
        prior = {"type": "uniform"}
    elif ia_prior == "breathe":
        prior = {"type": "custom", "p": get_IA_breathe_prior()}
    else:
        raise ValueError(f"ia_prior {ia_prior} not recognised")
    IA = VariableNode("Inactive alveoli (%)", 0, 30, 1, prior)

    # In reality O2 sat can't be below 70%.
    # However, the CPT should account for the fact that the lowest O2 sat is 82.8%.
    # 82.8-30 = 52.8%
    # TODO: should we hardcode the fact that the sum of AR and IA should not be below 70% O2 Sat?
    UO2Sat = VariableNode("Underlying O2 saturation (%)", 50, 100, 0.5, prior=None)
    O2Sat = VariableNode("O2 saturation (%)", 49.5, 100.5, 1, prior=None)

    # Set shared vars factor to node keys.
    # Used to aggregate messages up in longitudinal model
    key_hfev1 = f"['{ecFEV1.name}', '{HFEV1.name}', '{AR.name}'] -> {HFEV1.name}"
    key_ho2sat = f"['{O2SatFFA.name}', '{HO2Sat.name}', '{AR.name}'] -> {HO2Sat.name}"
    HFEV1.set_factor_to_node_key(key_hfev1)
    HO2Sat.set_factor_to_node_key(key_ho2sat)

    # Calculate CPTs
    ecFEV1.set_cpt(get_cpt([ecFEV1, HFEV1, AR]))
    O2SatFFA.set_cpt(get_cpt([O2SatFFA, HO2Sat, AR]))
    UO2Sat.set_cpt(get_cpt([UO2Sat, O2SatFFA, IA]))
    O2Sat.set_cpt(get_cpt([O2Sat, UO2Sat]))

    return HFEV1, ecFEV1, AR, HO2Sat, O2SatFFA, IA, UO2Sat, O2Sat


def o2sat_fev1_fef2575_point_in_time_model_shared_healthy_vars(
    height, age, sex, ia_prior="uniform", ar_prior="uniform"
):
    """
    Point in time model with full FEV1, FEF25-75 and O2Sat sides

    There is no factor linking AR and IA in this model. The priors for AR, IA are uniform
    """
    hfev1_prior = {"type": "default", "height": height, "age": age, "sex": sex}
    ho2sat_prior = {
        "type": "default",
        "height": height,
        "sex": sex,
    }
    HFEV1 = SharedVariableNode("Healthy FEV1 (L)", 1, 6, 0.05, prior=hfev1_prior)
    ecFEV1 = VariableNode("ecFEV1 (L)", 0, 6, 0.05, prior=None)
    ecFEF2575prctecFEV1 = VariableNode("ecFEF25-75 % ecFEV1 (%)", 0, 200, 2, prior=None)
    # Lowest predicted FEV1 is 15% (AR = 1-predictedFEV1)
    AR = VariableNode("Airway resistance (%)", 0, 90, 2, prior=None)
    if ar_prior == "uniform":
        AR.cpt = AR.set_prior({"type": "uniform"})
    elif ar_prior == "uniform in log space":
        AR.cpt = AR.set_prior(
            {"type": "custom", "p": get_uniform_prior_in_log_space(AR)}
        )
    elif ar_prior == "uniform message to HFEV1":
        AR.cpt = AR.set_prior(
            {"type": "custom", "p": get_prior_for_uniform_hfev1_message(AR)}
        )
    elif ar_prior == "breathe (2 days model, ecFEV1, ecFEF25-75)":
        AR.cpt = AR.set_prior(
            {
                "type": "custom",
                "p": ar.get_breathe_prior_from_2_days_model_ecFEV1_ecFEF2575(),
            }
        )
    elif ar_prior == "breathe (1 day model, O2Sat, ecFEV1)":
        AR.cpt = AR.set_prior(
            {
                "type": "custom",
                "p": ar.get_breathe_prior_from_1_day_model_o2sat_ecFEV1(),
            }
        )

    # Res 0.5 takes 19s, res 0.2 takes 21s
    HO2Sat = SharedVariableNode(
        "Healthy O2 saturation (%)", 90, 100, 0.5, prior=ho2sat_prior
    )
    # Highest drop is 92% (for AR = 90%)
    # Hence the lowest O2SatFFA is 90 * 0.92 = 82.8
    O2SatFFA = VariableNode(
        "O2 saturation if fully functional alveoli (%)", 80, 100, 0.5, prior=None
    )
    # O2 sat can't be below 70%.
    # If there's no airway resistance, it should still be possible to reach 70% O2 sat
    # Hence, min IA is 30% because i
    if ia_prior == "uniform":
        prior = {"type": "uniform"}
    elif ia_prior == "breathe":
        prior = {"type": "custom", "p": get_IA_breathe_prior()}
    else:
        raise ValueError(f"ia_prior {ia_prior} not recognised")
    IA = VariableNode("Inactive alveoli (%)", 0, 30, 1, prior=prior)

    # In reality O2 sat can't be below 70%.
    # However, the CPT should account for the fact that the lowest O2 sat is 82.8%.
    # 82.8-30 = 52.8%
    # TODO: should we hardcode the fact that the sum of AR and IA should not be below 70% O2 Sat?
    UO2Sat = VariableNode("Underlying O2 saturation (%)", 50, 100, 0.5, prior=None)
    O2Sat = VariableNode("O2 saturation (%)", 49.5, 100.5, 1, prior=None)

    # Set shared vars factor to node keys.
    # Used to aggregate messages up in longitudinal model
    key_hfev1 = f"['{ecFEV1.name}', '{HFEV1.name}', '{AR.name}'] -> {HFEV1.name}"
    key_ho2sat = f"['{O2SatFFA.name}', '{HO2Sat.name}', '{AR.name}'] -> {HO2Sat.name}"
    HFEV1.set_factor_to_node_key(key_hfev1)
    HO2Sat.set_factor_to_node_key(key_ho2sat)

    # Calculate CPTs
    ecFEV1.set_cpt(get_cpt([ecFEV1, HFEV1, AR]))
    O2SatFFA.set_cpt(get_cpt([O2SatFFA, HO2Sat, AR]))
    UO2Sat.set_cpt(get_cpt([UO2Sat, O2SatFFA, IA]))
    O2Sat.set_cpt(get_cpt([O2Sat, UO2Sat]))
    ecFEF2575prctecFEV1.set_cpt(get_cpt([ecFEF2575prctecFEV1, AR]))

    return HFEV1, ecFEV1, AR, HO2Sat, O2SatFFA, IA, UO2Sat, O2Sat, ecFEF2575prctecFEV1


def o2sat_fev1_fef2575_point_in_time_model_noise_shared_healthy_vars(
    height,
    age,
    sex,
    ia_prior="uniform",
    ar_prior="uniform",
    ecfev1_noise_model_cpt_suffix="_std_add_mult",
    ar_fef2575_cpt_suffix="_ecfev1_2_days_model_add_mult_noise",
):
    """
    Point in time model with full FEV1, FEF25-75 and O2Sat sides

    There is no factor linking AR and IA in this model. The priors for AR, IA are uniform
    """
    hfev1_prior = {"type": "default", "height": height, "age": age, "sex": sex}
    ho2sat_prior = {
        "type": "default",
        "height": height,
        "sex": sex,
    }
    HFEV1 = SharedVariableNode("Healthy FEV1 (L)", 1, 6, 0.05, prior=hfev1_prior)

    ecFEV1 = VariableNode("ecFEV1 (L)", 0, 6, 0.05, prior=None)
    uecFEV1 = VariableNode("Underlying ecFEV1 (L)", 0, 6, 0.05, prior=None)

    ecFEF2575prctecFEV1 = VariableNode("ecFEF25-75 % ecFEV1 (%)", 0, 200, 2, prior=None)
    # Lowest predicted FEV1 is 15% (AR = 1-predictedFEV1)
    AR = VariableNode("Airway resistance (%)", 0, 90, 2, prior=None)
    if ar_prior == "uniform":
        AR.cpt = AR.set_prior({"type": "uniform"})
    elif ar_prior == "uniform in log space":
        AR.cpt = AR.set_prior(
            {"type": "custom", "p": get_uniform_prior_in_log_space(AR)}
        )
    elif ar_prior == "uniform message to HFEV1":
        AR.cpt = AR.set_prior(
            {"type": "custom", "p": get_prior_for_uniform_hfev1_message(AR)}
        )
    elif ar_prior == "breathe (2 days model, ecFEV1, ecFEF25-75)":
        AR.cpt = AR.set_prior(
            {
                "type": "custom",
                "p": ar.get_breathe_prior_from_2_days_model_ecFEV1_ecFEF2575(),
            }
        )
    elif ar_prior == "breathe (1 day model, O2Sat, ecFEV1)":
        AR.cpt = AR.set_prior(
            {
                "type": "custom",
                "p": ar.get_breathe_prior_from_1_day_model_o2sat_ecFEV1(),
            }
        )
    elif ar_prior == "breathe (2 days model, ecFEV1, ecFEF25-75, add mult noise)":
        AR.cpt = AR.set_prior(
            {
                "type": "custom",
                "p": ar.get_breathe_prior_from_2_days_model_ecFEV1_ecFEF2575_ecfev1addmultnoise(),
            }
        )

    # Res 0.5 takes 19s, res 0.2 takes 21s
    HO2Sat = SharedVariableNode(
        "Healthy O2 saturation (%)", 90, 100, 0.5, prior=ho2sat_prior
    )
    # Highest drop is 92% (for AR = 90%)
    # Hence the lowest O2SatFFA is 90 * 0.92 = 82.8
    O2SatFFA = VariableNode(
        "O2 saturation if fully functional alveoli (%)", 80, 100, 0.5, prior=None
    )
    # O2 sat can't be below 70%.
    # If there's no airway resistance, it should still be possible to reach 70% O2 sat
    # Hence, min IA is 30% because i
    if ia_prior == "uniform":
        prior = {"type": "uniform"}
    elif ia_prior == "breathe":
        prior = {"type": "custom", "p": get_IA_breathe_prior()}
    else:
        raise ValueError(f"ia_prior {ia_prior} not recognised")
    IA = VariableNode("Inactive alveoli (%)", 0, 30, 1, prior=prior)

    # In reality O2 sat can't be below 70%.
    # However, the CPT should account for the fact that the lowest O2 sat is 82.8%.
    # 82.8-30 = 52.8%
    # TODO: should we hardcode the fact that the sum of AR and IA should not be below 70% O2 Sat?
    UO2Sat = VariableNode("Underlying O2 saturation (%)", 50, 100, 0.5, prior=None)
    O2Sat = VariableNode("O2 saturation (%)", 49.5, 100.5, 1, prior=None)

    # Set shared vars factor to node keys.
    # Used to aggregate messages up in longitudinal model
    key_hfev1 = f"['{uecFEV1.name}', '{HFEV1.name}', '{AR.name}'] -> {HFEV1.name}"
    key_ho2sat = f"['{O2SatFFA.name}', '{HO2Sat.name}', '{AR.name}'] -> {HO2Sat.name}"
    HFEV1.set_factor_to_node_key(key_hfev1)
    HO2Sat.set_factor_to_node_key(key_ho2sat)

    # Calculate CPTs
    ecFEV1.set_cpt(get_cpt([ecFEV1, uecFEV1], suffix=ecfev1_noise_model_cpt_suffix))
    uecFEV1.set_cpt(get_cpt([uecFEV1, HFEV1, AR]))
    O2SatFFA.set_cpt(get_cpt([O2SatFFA, HO2Sat, AR]))
    UO2Sat.set_cpt(get_cpt([UO2Sat, O2SatFFA, IA]))
    O2Sat.set_cpt(get_cpt([O2Sat, UO2Sat]))
    ecFEF2575prctecFEV1.set_cpt(
        get_cpt([ecFEF2575prctecFEV1, AR], suffix=ar_fef2575_cpt_suffix)
    )

    return (
        HFEV1,
        uecFEV1,
        ecFEV1,
        AR,
        HO2Sat,
        O2SatFFA,
        IA,
        UO2Sat,
        O2Sat,
        ecFEF2575prctecFEV1,
    )


def o2sat_fev1_fef2575_point_in_time_model_noise_shared_healthy_vars_log(
    height,
    age,
    sex,
    ia_prior="uniform",
    ar_prior="uniform",
    ecfev1_noise_model_cpt_suffix="_std_add_mult",
    ar_fef2575_cpt_suffix="_ecfev1_2_days_model_add_mult_noise",
):
    """
    Point in time model with full FEV1, FEF25-75 and O2Sat sides

    There is no factor linking AR and IA in this model. The priors for AR, IA are uniform
    """
    hfev1_prior = {"type": "default", "height": height, "age": age, "sex": sex}
    ho2sat_prior = {
        "type": "default",
        "height": height,
        "sex": sex,
    }
    HFEV1 = SharedVariableNode("Healthy FEV1 (L)", 0, 6, 0.05, prior=hfev1_prior)
    HFEV1.cpt = get_p_in_log(HFEV1, HFEV1.cpt)

    ecFEV1 = VariableNode("ecFEV1 (L)", 0, 6, 0.05, prior=None)
    uecFEV1 = VariableNode("Underlying ecFEV1 (L)", 0, 6, 0.05, prior=None)

    ecFEF2575prctecFEV1 = VariableNode("ecFEF25-75 % ecFEV1 (%)", 0, 200, 2, prior=None)
    # Lowest predicted FEV1 is 15% (AR = 1-predictedFEV1)
    AR = VariableNode("Airway resistance (%)", 0, 90, 2, prior=None)
    if ar_prior == "uniform":
        AR.cpt = AR.set_prior({"type": "uniform"})
    elif ar_prior == "uniform in log space":
        AR.cpt = AR.set_prior(
            {"type": "custom", "p": get_uniform_prior_in_log_space(AR)}
        )
    elif ar_prior == "uniform message to HFEV1":
        AR.cpt = AR.set_prior(
            {"type": "custom", "p": get_prior_for_uniform_hfev1_message(AR)}
        )
    elif ar_prior == "breathe (2 days model, ecFEV1, ecFEF25-75)":
        AR.cpt = AR.set_prior(
            {
                "type": "custom",
                "p": ar.get_breathe_prior_from_2_days_model_ecFEV1_ecFEF2575(),
            }
        )
    elif ar_prior == "breathe (1 day model, O2Sat, ecFEV1)":
        AR.cpt = AR.set_prior(
            {
                "type": "custom",
                "p": ar.get_breathe_prior_from_1_day_model_o2sat_ecFEV1(),
            }
        )
    elif ar_prior == "breathe (2 days model, ecFEV1, ecFEF25-75, add mult noise)":
        AR.cpt = AR.set_prior(
            {
                "type": "custom",
                "p": ar.get_breathe_prior_from_2_days_model_ecFEV1_ecFEF2575_ecfev1addmultnoise(),
            }
        )

    # Res 0.5 takes 19s, res 0.2 takes 21s
    HO2Sat = SharedVariableNode(
        "Healthy O2 saturation (%)", 90, 100, 0.5, prior=ho2sat_prior
    )
    # Highest drop is 92% (for AR = 90%)
    # Hence the lowest O2SatFFA is 90 * 0.92 = 82.8
    O2SatFFA = VariableNode(
        "O2 saturation if fully functional alveoli (%)", 80, 100, 0.5, prior=None
    )
    # O2 sat can't be below 70%.
    # If there's no airway resistance, it should still be possible to reach 70% O2 sat
    # Hence, min IA is 30% because i
    if ia_prior == "uniform":
        prior = {"type": "uniform"}
    elif ia_prior == "breathe":
        prior = {"type": "custom", "p": get_IA_breathe_prior()}
    else:
        raise ValueError(f"ia_prior {ia_prior} not recognised")
    IA = VariableNode("Inactive alveoli (%)", 0, 30, 1, prior=prior)

    # In reality O2 sat can't be below 70%.
    # However, the CPT should account for the fact that the lowest O2 sat is 82.8%.
    # 82.8-30 = 52.8%
    # TODO: should we hardcode the fact that the sum of AR and IA should not be below 70% O2 Sat?
    UO2Sat = VariableNode("Underlying O2 saturation (%)", 50, 100, 0.5, prior=None)
    O2Sat = VariableNode("O2 saturation (%)", 49.5, 100.5, 1, prior=None)

    # Set shared vars factor to node keys.
    # Used to aggregate messages up in longitudinal model
    key_hfev1 = f"['{uecFEV1.name}', '{HFEV1.name}', '{AR.name}'] -> {HFEV1.name}"
    key_ho2sat = f"['{O2SatFFA.name}', '{HO2Sat.name}', '{AR.name}'] -> {HO2Sat.name}"
    HFEV1.set_factor_to_node_key(key_hfev1)
    HO2Sat.set_factor_to_node_key(key_ho2sat)

    # Calculate CPTs
    ecFEV1.set_cpt(get_cpt([ecFEV1, uecFEV1], suffix=ecfev1_noise_model_cpt_suffix))
    uecFEV1.set_cpt(get_cpt([uecFEV1, HFEV1, AR]))
    O2SatFFA.set_cpt(get_cpt([O2SatFFA, HO2Sat, AR]))
    UO2Sat.set_cpt(get_cpt([UO2Sat, O2SatFFA, IA]))
    O2Sat.set_cpt(get_cpt([O2Sat, UO2Sat]))
    ecFEF2575prctecFEV1.set_cpt(
        get_cpt([ecFEF2575prctecFEV1, AR], suffix=ar_fef2575_cpt_suffix)
    )

    return (
        HFEV1,
        uecFEV1,
        ecFEV1,
        AR,
        HO2Sat,
        O2SatFFA,
        IA,
        UO2Sat,
        O2Sat,
        ecFEF2575prctecFEV1,
    )


def o2sat_fev1_fef2575_point_in_time_model_noise_shared_healthy_vars_light(
    height,
    age,
    sex,
    ia_prior="uniform",
    ar_prior="uniform",
    ecfev1_noise_model_cpt_suffix="_std_add_mult",
):
    """
    Point in time model with full FEV1, FEF25-75 and O2Sat sides

    There is no factor linking AR and IA in this model. The priors for AR, IA are uniform
    """
    hfev1_prior = {"type": "default", "height": height, "age": age, "sex": sex}
    ho2sat_prior = {
        "type": "default",
        "height": height,
        "sex": sex,
    }
    HFEV1 = SharedVariableNode("Healthy FEV1 (L)", 1, 6, 1, prior=hfev1_prior)
    # HFEV1_point_mass_prior = np.zeros(HFEV1.card)
    # idx_three_point_five = HFEV1.get_bin_for_value(3.5)[1]
    # HFEV1_point_mass_prior[idx_three_point_five] = 1
    # HFEV1.cpt = HFEV1.set_prior({"type": "custom", "p": HFEV1_point_mass_prior})

    ecFEV1 = VariableNode("ecFEV1 (L)", 0, 6, 1, prior=None)
    uecFEV1 = VariableNode("Underlying ecFEV1 (L)", 0, 6, 1, prior=None)

    ecFEF2575prctecFEV1 = VariableNode(
        "ecFEF25-75 % ecFEV1 (%)", 0, 200, 20, prior=None
    )

    # Lowest predicted FEV1 is 15% (AR = 1-predictedFEV1)
    AR = VariableNode("Airway resistance (%)", 0, 90, 10, prior=None)
    if ar_prior == "uniform":
        AR.cpt = AR.set_prior({"type": "uniform"})
    elif ar_prior == "uniform in log space":
        AR.cpt = AR.set_prior(
            {"type": "custom", "p": get_uniform_prior_in_log_space(AR)}
        )
    elif ar_prior == "uniform message to HFEV1":
        AR.cpt = AR.set_prior(
            {"type": "custom", "p": get_prior_for_uniform_hfev1_message(AR)}
        )
    elif ar_prior == "breathe (2 days model, ecFEV1, ecFEF25-75)":
        AR.cpt = AR.set_prior(
            {
                "type": "custom",
                "p": ar.get_breathe_prior_from_2_days_model_ecFEV1_ecFEF2575(),
            }
        )
    elif ar_prior == "breathe (1 day model, O2Sat, ecFEV1)":
        AR.cpt = AR.set_prior(
            {
                "type": "custom",
                "p": ar.get_breathe_prior_from_1_day_model_o2sat_ecFEV1(),
            }
        )

    # Res 0.5 takes 19s, res 0.2 takes 21s
    HO2Sat = SharedVariableNode(
        "Healthy O2 saturation (%)", 90, 100, 2, prior=ho2sat_prior
    )
    # Highest drop is 92% (for AR = 90%)
    # Hence the lowest O2SatFFA is 90 * 0.92 = 82.8
    O2SatFFA = VariableNode(
        "O2 saturation if fully functional alveoli (%)", 80, 100, 2, prior=None
    )
    # O2 sat can't be below 70%.
    # If there's no airway resistance, it should still be possible to reach 70% O2 sat
    # Hence, min IA is 30% because i
    if ia_prior == "uniform":
        prior = {"type": "uniform"}
    elif ia_prior == "breathe":
        prior = {"type": "custom", "p": get_IA_breathe_prior()}
    else:
        raise ValueError(f"ia_prior {ia_prior} not recognised")
    IA = VariableNode("Inactive alveoli (%)", 0, 30, 2, prior=prior)

    # In reality O2 sat can't be below 70%.
    # However, the CPT should account for the fact that the lowest O2 sat is 82.8%.
    # 82.8-30 = 52.8%
    # TODO: should we hardcode the fact that the sum of AR and IA should not be below 70% O2 Sat?
    UO2Sat = VariableNode("Underlying O2 saturation (%)", 50, 100, 2, prior=None)
    bin_width = 2
    O2Sat = VariableNode(
        "O2 saturation (%)",
        50 - bin_width / 2,
        100 + bin_width / 2,
        bin_width,
        prior=None,
    )

    # Set shared vars factor to node keys.
    # Used to aggregate messages up in longitudinal model
    key_hfev1 = f"['{uecFEV1.name}', '{HFEV1.name}', '{AR.name}'] -> {HFEV1.name}"
    key_ho2sat = f"['{O2SatFFA.name}', '{HO2Sat.name}', '{AR.name}'] -> {HO2Sat.name}"
    HFEV1.set_factor_to_node_key(key_hfev1)
    HO2Sat.set_factor_to_node_key(key_ho2sat)

    # Calculate CPTs
    uecFEV1.set_cpt(get_cpt([uecFEV1, HFEV1, AR]))
    ecFEV1.set_cpt(get_cpt([ecFEV1, uecFEV1], suffix=ecfev1_noise_model_cpt_suffix))
    ecFEF2575prctecFEV1.set_cpt(get_cpt([ecFEF2575prctecFEV1, AR]))
    O2SatFFA.set_cpt(get_cpt([O2SatFFA, HO2Sat, AR]))
    UO2Sat.set_cpt(get_cpt([UO2Sat, O2SatFFA, IA]))
    O2Sat.set_cpt(get_cpt([O2Sat, UO2Sat]))

    return (
        HFEV1,
        uecFEV1,
        ecFEV1,
        AR,
        HO2Sat,
        O2SatFFA,
        IA,
        UO2Sat,
        O2Sat,
        ecFEF2575prctecFEV1,
    )


def o2sat_fev1_point_in_time_model_shared_healthy_vars_light(
    height, age, sex, ia_prior="uniform"
):
    """
    Point in time model with full FEV1, FEF25-75 and O2Sat sides

    There is no factor linking AR and IA in this model. The priors for AR, IA are uniform

    The model is light (ex: HFEV1, ecFEV1 has 6 states instead of 120)
    """
    hfev1_prior = {"type": "default", "height": height, "age": age, "sex": sex}
    ho2sat_prior = {
        "type": "default",
        "height": height,
        "sex": sex,
    }
    HFEV1 = SharedVariableNode("Healthy FEV1 (L)", 1, 6, 1, prior=hfev1_prior)
    ecFEV1 = VariableNode("ecFEV1 (L)", 0, 6, 1, prior=None)
    # Lowest predicted FEV1 is 15% (AR = 1-predictedFEV1)
    AR = VariableNode("Airway resistance (%)", 0, 90, 10, prior={"type": "uniform"})

    # Res 0.5 takes 19s, res 0.2 takes 21s
    HO2Sat = SharedVariableNode(
        "Healthy O2 saturation (%)", 90, 100, 2, prior=ho2sat_prior
    )
    # Highest drop is 92% (for AR = 90%)
    # Hence the lowest O2SatFFA is 90 * 0.92 = 82.8
    O2SatFFA = VariableNode(
        "O2 saturation if fully functional alveoli (%)", 80, 100, 2, prior=None
    )
    # O2 sat can't be below 70%.
    # If there's no airway resistance, it should still be possible to reach 70% O2 sat
    # Hence, min IA is 30% because i
    if ia_prior == "uniform":
        prior = {"type": "uniform"}
    elif ia_prior == "breathe":
        prior = {"type": "custom", "p": get_IA_breathe_prior()}
    else:
        raise ValueError(f"ia_prior {ia_prior} not recognised")
    IA = VariableNode("Inactive alveoli (%)", 0, 30, 2, prior=prior)

    # In reality O2 sat can't be below 70%.
    # However, the CPT should account for the fact that the lowest O2 sat is 82.8%.
    # 82.8-30 = 52.8%
    # TODO: should we hardcode the fact that the sum of AR and IA should not be below 70% O2 Sat?
    UO2Sat = VariableNode("Underlying O2 saturation (%)", 50, 100, 2, prior=None)
    bin_width = 2
    O2Sat = VariableNode(
        "O2 saturation (%)",
        50 - bin_width / 2,
        100 + bin_width / 2,
        bin_width,
        prior=None,
    )

    # Set shared vars factor to node keys.
    # Used to aggregate messages up in longitudinal model
    key_hfev1 = f"['{ecFEV1.name}', '{HFEV1.name}', '{AR.name}'] -> {HFEV1.name}"
    key_ho2sat = f"['{O2SatFFA.name}', '{HO2Sat.name}', '{AR.name}'] -> {HO2Sat.name}"
    HFEV1.set_factor_to_node_key(key_hfev1)
    HO2Sat.set_factor_to_node_key(key_ho2sat)

    # Calculate CPTs
    ecFEV1.set_cpt(get_cpt([ecFEV1, HFEV1, AR]))
    O2SatFFA.set_cpt(get_cpt([O2SatFFA, HO2Sat, AR]))
    UO2Sat.set_cpt(get_cpt([UO2Sat, O2SatFFA, IA]))
    O2Sat.set_cpt(get_cpt([O2Sat, UO2Sat]))

    return (
        HFEV1,
        ecFEV1,
        AR,
        HO2Sat,
        O2SatFFA,
        IA,
        UO2Sat,
        O2Sat,
    )


def o2sat_fev1_fef2575_long_model_shared_healthy_vars_and_temporal_ar(
    height,
    age,
    sex,
    ia_prior="uniform",
    ar_prior="uniform",
    ar_change_cpt_suffix="",
    n_cutset_conditioned_states=None,
):
    """
    Longitudinal model with full FEV1, FEF25-75 and O2Sat sides
    The airway resistances has day-to-day temporal connection
    """
    hfev1_prior = {"type": "default", "height": height, "age": age, "sex": sex}
    ho2sat_prior = {
        "type": "default",
        "height": height,
        "sex": sex,
    }
    HFEV1 = SharedVariableNode("Healthy FEV1 (L)", 1, 6, 0.05, prior=hfev1_prior)
    ecFEV1 = VariableNode("ecFEV1 (L)", 0, 6, 0.05, prior=None)
    ecFEF2575prctecFEV1 = VariableNode("ecFEF25-75 % ecFEV1 (%)", 0, 200, 2, prior=None)
    # Lowest predicted FEV1 is 15% (AR = 1-predictedFEV1)

    DE = DiscreteVariableNode("Days elapsed", 1, 3, 1)

    if n_cutset_conditioned_states is not None:
        AR = CutsetConditionedTemporalVariableNode(
            "Airway resistance (%)", 0, 90, 2, n_cutset_conditioned_states
        )
    else:
        AR = TemporalVariableNode("Airway resistance (%)", 0, 90, 2)

    AR.set_change_cpt(get_cpt([AR, AR, DE], suffix=ar_change_cpt_suffix))

    if ar_prior == "uniform":
        AR.set_first_day_prior({"type": "uniform"})
    elif ar_prior == "uniform in log space":
        AR.set_first_day_prior(
            {"type": "custom", "p": get_uniform_prior_in_log_space(AR)}
        )
    elif ar_prior == "uniform message to HFEV1":
        AR.set_first_day_prior(
            {"type": "custom", "p": get_prior_for_uniform_hfev1_message(AR)}
        )
    elif ar_prior == "breathe (2 days model, ecFEV1, ecFEF25-75)":
        AR.set_first_day_prior(
            {
                "type": "custom",
                "p": ar.get_breathe_prior_from_2_days_model_ecFEV1_ecFEF2575(),
            }
        )
    elif ar_prior == "breathe (1 day model, O2Sat, ecFEV1)":
        AR.set_first_day_prior(
            {
                "type": "custom",
                "p": ar.get_breathe_prior_from_1_day_model_o2sat_ecFEV1(),
            }
        )

    # Res 0.5 takes 19s, res 0.2 takes 21s
    HO2Sat = SharedVariableNode(
        "Healthy O2 saturation (%)", 90, 100, 0.5, prior=ho2sat_prior
    )
    # Highest drop is 92% (for AR = 90%)
    # Hence the lowest O2SatFFA is 90 * 0.92 = 82.8
    O2SatFFA = VariableNode(
        "O2 saturation if fully functional alveoli (%)", 80, 100, 0.5, prior=None
    )
    # O2 sat can't be below 70%.
    # If there's no airway resistance, it should still be possible to reach 70% O2 sat
    # Hence, min IA is 30% because i
    if ia_prior == "uniform":
        prior = {"type": "uniform"}
    elif ia_prior == "breathe":
        prior = {"type": "custom", "p": get_IA_breathe_prior()}
    else:
        raise ValueError(f"ia_prior {ia_prior} not recognised")
    IA = VariableNode("Inactive alveoli (%)", 0, 30, 1, prior=prior)

    # In reality O2 sat can't be below 70%.
    # However, the CPT should account for the fact that the lowest O2 sat is 82.8%.
    # 82.8-30 = 52.8%
    # TODO: should we hardcode the fact that the sum of AR and IA should not be below 70% O2 Sat?
    UO2Sat = VariableNode("Underlying O2 saturation (%)", 50, 100, 0.5, prior=None)
    O2Sat = VariableNode("O2 saturation (%)", 49.5, 100.5, 1, prior=None)

    # Set shared vars factor to node keys.
    # Used to aggregate messages up in longitudinal model
    key_hfev1 = f"['{ecFEV1.name}', '{HFEV1.name}', '{AR.name}'] -> {HFEV1.name}"
    key_ho2sat = f"['{O2SatFFA.name}', '{HO2Sat.name}', '{AR.name}'] -> {HO2Sat.name}"
    HFEV1.set_factor_to_node_key(key_hfev1)
    HO2Sat.set_factor_to_node_key(key_ho2sat)

    # Calculate CPTs
    ecFEV1.set_cpt(get_cpt([ecFEV1, HFEV1, AR]))
    O2SatFFA.set_cpt(get_cpt([O2SatFFA, HO2Sat, AR]))
    UO2Sat.set_cpt(get_cpt([UO2Sat, O2SatFFA, IA]))
    O2Sat.set_cpt(get_cpt([O2Sat, UO2Sat]))
    ecFEF2575prctecFEV1.set_cpt(get_cpt([ecFEF2575prctecFEV1, AR]))

    return HFEV1, ecFEV1, AR, HO2Sat, O2SatFFA, IA, UO2Sat, O2Sat, ecFEF2575prctecFEV1


def o2sat_fev1_fef2575_long_model_noise_shared_healthy_vars_and_temporal_ar(
    height,
    age,
    sex,
    ia_prior="uniform",
    ar_prior="uniform",
    ar_change_cpt_suffix=None,
    n_cutset_conditioned_states=None,
    ecfev1_noise_model_suffix=None,
    fef2575_cpt_suffix=None,
):
    """
    Longitudinal model with full FEV1, FEF25-75 and O2Sat sides
    The airway resistances has day-to-day temporal connection
    """
    hfev1_prior = {"type": "default", "height": height, "age": age, "sex": sex}
    ho2sat_prior = {
        "type": "default",
        "height": height,
        "sex": sex,
    }
    HFEV1 = SharedVariableNode("Healthy FEV1 (L)", 1, 6, 0.05, prior=hfev1_prior)
    # HFEV1_point_mass_prior = np.zeros(HFEV1.card)
    # idx_three_point_five = HFEV1.get_bin_for_value(3.5)[1]
    # HFEV1_point_mass_prior[idx_three_point_five] = 1
    # HFEV1.cpt = HFEV1.set_prior({"type": "custom", "p": HFEV1_point_mass_prior})

    uecFEV1 = VariableNode("Underlying ecFEV1 (L)", 0, 6, 0.05, prior=None)
    ecFEV1 = VariableNode("ecFEV1 (L)", 0, 6, 0.05, prior=None)
    ecFEF2575prctecFEV1 = VariableNode("ecFEF25-75 % ecFEV1 (%)", 0, 200, 2, prior=None)
    # Lowest predicted FEV1 is 15% (AR = 1-predictedFEV1)

    if n_cutset_conditioned_states is not None:
        AR = CutsetConditionedTemporalVariableNode(
            "Airway resistance (%)", 0, 90, 2, n_cutset_conditioned_states
        )
    else:
        AR = TemporalVariableNode("Airway resistance (%)", 0, 90, 2)

    # Select change CPT depending on the suffix
    if ar_change_cpt_suffix == "_shape_factor10":
        S = DiscreteVariableNode("AR change factor shape", 1, 10, 1)
        AR.set_change_cpt(get_cpt([AR, AR, S], suffix=ar_change_cpt_suffix))
    elif ar_change_cpt_suffix == "_shape_factor27":
        S = DiscreteVariableNode("AR change factor shape", 1, 27, 1)
        AR.set_change_cpt(get_cpt([AR, AR, S], suffix=ar_change_cpt_suffix))
    elif ar_change_cpt_suffix == "_shape_factor3":
        S = DiscreteVariableNode("AR change factor shape", 1, 3, 1)
        AR.set_change_cpt(get_cpt([AR, AR, S], suffix=ar_change_cpt_suffix))
    else:
        DE = DiscreteVariableNode("Days elapsed", 1, 3, 1)
        AR.set_change_cpt(get_cpt([AR, AR, DE], suffix=ar_change_cpt_suffix))

    if ar_prior == "uniform":
        AR.set_first_day_prior({"type": "uniform"})
    elif ar_prior == "uniform in log space":
        AR.set_first_day_prior(
            {"type": "custom", "p": get_uniform_prior_in_log_space(AR)}
        )
    elif ar_prior == "uniform message to HFEV1":
        AR.set_first_day_prior(
            {"type": "custom", "p": get_prior_for_uniform_hfev1_message(AR)}
        )
    elif ar_prior == "breathe (2 days model, ecFEV1, ecFEF25-75)":
        AR.set_first_day_prior(
            {
                "type": "custom",
                "p": ar.get_breathe_prior_from_2_days_model_ecFEV1_ecFEF2575(),
            }
        )
    elif ar_prior == "breathe (1 day model, O2Sat, ecFEV1)":
        AR.set_first_day_prior(
            {
                "type": "custom",
                "p": ar.get_breathe_prior_from_1_day_model_o2sat_ecFEV1(),
            }
        )
    elif ar_prior == "breathe (2 days model, ecFEV1 addmultnoise, ecFEF25-75)":
        AR.set_first_day_prior(
            {
                "type": "custom",
                "p": ar.get_breathe_prior_from_2_days_model_ecFEV1_ecFEF2575_ecfev1addmultnoise(),
            }
        )

    # Res 0.5 takes 19s, res 0.2 takes 21s
    HO2Sat = SharedVariableNode(
        "Healthy O2 saturation (%)", 90, 100, 0.5, prior=ho2sat_prior
    )
    # Highest drop is 92% (for AR = 90%)
    # Hence the lowest O2SatFFA is 90 * 0.92 = 82.8
    O2SatFFA = VariableNode(
        "O2 saturation if fully functional alveoli (%)", 80, 100, 0.5, prior=None
    )
    # O2 sat can't be below 70%.
    # If there's no airway resistance, it should still be possible to reach 70% O2 sat
    # Hence, min IA is 30% because i
    if ia_prior == "uniform":
        prior = {"type": "uniform"}
    elif ia_prior == "breathe":
        prior = {"type": "custom", "p": get_IA_breathe_prior()}
    else:
        raise ValueError(f"ia_prior {ia_prior} not recognised")
    IA = VariableNode("Inactive alveoli (%)", 0, 30, 1, prior=prior)

    # In reality O2 sat can't be below 70%.
    # However, the CPT should account for the fact that the lowest O2 sat is 82.8%.
    # 82.8-30 = 52.8%
    # TODO: should we hardcode the fact that the sum of AR and IA should not be below 70% O2 Sat?
    UO2Sat = VariableNode("Underlying O2 saturation (%)", 50, 100, 0.5, prior=None)
    bin_width = 1
    O2Sat = VariableNode(
        "O2 saturation (%)",
        50 - bin_width / 2,
        100 + bin_width / 2,
        bin_width,
        prior=None,
    )

    # Set shared vars factor to node keys.
    # Used to aggregate messages up in longitudinal model
    key_hfev1 = f"['{uecFEV1.name}', '{HFEV1.name}', '{AR.name}'] -> {HFEV1.name}"
    key_ho2sat = f"['{O2SatFFA.name}', '{HO2Sat.name}', '{AR.name}'] -> {HO2Sat.name}"
    HFEV1.set_factor_to_node_key(key_hfev1)
    HO2Sat.set_factor_to_node_key(key_ho2sat)

    # Calculate CPTs
    uecFEV1.set_cpt(get_cpt([uecFEV1, HFEV1, AR]))
    ecFEV1.set_cpt(get_cpt([ecFEV1, uecFEV1], suffix=ecfev1_noise_model_suffix))
    O2SatFFA.set_cpt(get_cpt([O2SatFFA, HO2Sat, AR]))
    UO2Sat.set_cpt(get_cpt([UO2Sat, O2SatFFA, IA]))
    O2Sat.set_cpt(get_cpt([O2Sat, UO2Sat]))
    ecFEF2575prctecFEV1.set_cpt(
        get_cpt([ecFEF2575prctecFEV1, AR], suffix=fef2575_cpt_suffix)
    )

    return (
        HFEV1,
        uecFEV1,
        ecFEV1,
        AR,
        HO2Sat,
        O2SatFFA,
        IA,
        UO2Sat,
        O2Sat,
        ecFEF2575prctecFEV1,
    )


def o2sat_fev1_fef2575_long_model_noise_shared_healthy_vars_and_temporal_ar_light(
    height,
    age,
    sex,
    ia_prior="uniform",
    ar_prior="uniform",
    ar_change_cpt_suffix="",
    n_cutset_conditioned_states=None,
):
    """
    Longitudinal model with full FEV1, FEF25-75 and O2Sat sides
    The airway resistances has day-to-day temporal connection

    FEV1 noise model suffix fixed to 0.7, high noise to compensate low granularity for the temporal ARs
    """
    hfev1_prior = {"type": "default", "height": height, "age": age, "sex": sex}
    ho2sat_prior = {
        "type": "default",
        "height": height,
        "sex": sex,
    }
    HFEV1 = SharedVariableNode("Healthy FEV1 (L)", 1, 6, 1, prior=hfev1_prior)
    # HFEV1_point_mass_prior = np.zeros(HFEV1.card)
    # idx_three_point_five = HFEV1.get_bin_for_value(3.5)[1]
    # HFEV1_point_mass_prior[idx_three_point_five] = 1
    # HFEV1.cpt = HFEV1.set_prior({"type": "custom", "p": HFEV1_point_mass_prior})

    uecFEV1 = VariableNode("Underlying ecFEV1 (L)", 0, 6, 1, prior=None)
    ecFEV1 = VariableNode("ecFEV1 (L)", 0, 6, 1, prior=None)
    ecFEF2575prctecFEV1 = VariableNode(
        "ecFEF25-75 % ecFEV1 (%)", 0, 200, 20, prior=None
    )
    # Lowest predicted FEV1 is 15% (AR = 1-predictedFEV1)

    if n_cutset_conditioned_states is not None:
        AR = CutsetConditionedTemporalVariableNode(
            "Airway resistance (%)", 0, 90, 10, n_cutset_conditioned_states
        )
    else:
        AR = TemporalVariableNode("Airway resistance (%)", 0, 90, 10)

    if ar_change_cpt_suffix == "_shape_factor":
        S = DiscreteVariableNode("AR change factor shape", 2, 10, 2)
        AR.set_change_cpt(get_cpt([AR, AR, S], suffix=ar_change_cpt_suffix))
    else:
        DE = DiscreteVariableNode("Days elapsed", 1, 3, 1)
        AR.set_change_cpt(get_cpt([AR, AR, DE], suffix=ar_change_cpt_suffix))

    if ar_prior == "uniform":
        AR.set_first_day_prior({"type": "uniform"})
    elif ar_prior == "uniform in log space":
        AR.set_first_day_prior(
            {"type": "custom", "p": get_uniform_prior_in_log_space(AR)}
        )
    elif ar_prior == "uniform message to HFEV1":
        AR.set_first_day_prior(
            {"type": "custom", "p": get_prior_for_uniform_hfev1_message(AR)}
        )
    elif ar_prior == "breathe (2 days model, ecFEV1, ecFEF25-75)":
        AR.set_first_day_prior(
            {
                "type": "custom",
                "p": ar.get_breathe_prior_from_2_days_model_ecFEV1_ecFEF2575(),
            }
        )
    elif ar_prior == "breathe (1 day model, O2Sat, ecFEV1)":
        AR.set_first_day_prior(
            {
                "type": "custom",
                "p": ar.get_breathe_prior_from_1_day_model_o2sat_ecFEV1(),
            }
        )

    # Res 0.5 takes 19s, res 0.2 takes 21s
    HO2Sat = SharedVariableNode(
        "Healthy O2 saturation (%)", 90, 100, 2, prior=ho2sat_prior
    )
    # Highest drop is 92% (for AR = 90%)
    # Hence the lowest O2SatFFA is 90 * 0.92 = 82.8
    O2SatFFA = VariableNode(
        "O2 saturation if fully functional alveoli (%)", 80, 100, 2, prior=None
    )
    # O2 sat can't be below 70%.
    # If there's no airway resistance, it should still be possible to reach 70% O2 sat
    # Hence, min IA is 30% because i
    if ia_prior == "uniform":
        prior = {"type": "uniform"}
    elif ia_prior == "breathe":
        prior = {"type": "custom", "p": get_IA_breathe_prior()}
    else:
        raise ValueError(f"ia_prior {ia_prior} not recognised")
    IA = VariableNode("Inactive alveoli (%)", 0, 30, 2, prior=prior)

    # In reality O2 sat can't be below 70%.
    # However, the CPT should account for the fact that the lowest O2 sat is 82.8%.
    # 82.8-30 = 52.8%
    # TODO: should we hardcode the fact that the sum of AR and IA should not be below 70% O2 Sat?
    UO2Sat = VariableNode("Underlying O2 saturation (%)", 50, 100, 2, prior=None)
    bin_width = 2
    O2Sat = VariableNode(
        "O2 saturation (%)",
        50 - bin_width / 2,
        100 + bin_width / 2,
        bin_width,
        prior=None,
    )

    # Set shared vars factor to node keys.
    # Used to aggregate messages up in longitudinal model
    key_hfev1 = f"['{uecFEV1.name}', '{HFEV1.name}', '{AR.name}'] -> {HFEV1.name}"
    key_ho2sat = f"['{O2SatFFA.name}', '{HO2Sat.name}', '{AR.name}'] -> {HO2Sat.name}"
    HFEV1.set_factor_to_node_key(key_hfev1)
    HO2Sat.set_factor_to_node_key(key_ho2sat)

    # Calculate CPTs
    uecFEV1.set_cpt(get_cpt([uecFEV1, HFEV1, AR]))
    ecFEV1.set_cpt(get_cpt([ecFEV1, uecFEV1], suffix="_std_0.7"))
    O2SatFFA.set_cpt(get_cpt([O2SatFFA, HO2Sat, AR]))
    UO2Sat.set_cpt(get_cpt([UO2Sat, O2SatFFA, IA]))
    O2Sat.set_cpt(get_cpt([O2Sat, UO2Sat]))
    ecFEF2575prctecFEV1.set_cpt(get_cpt([ecFEF2575prctecFEV1, AR]))

    return (
        HFEV1,
        uecFEV1,
        ecFEV1,
        AR,
        HO2Sat,
        O2SatFFA,
        IA,
        UO2Sat,
        O2Sat,
        ecFEF2575prctecFEV1,
    )
