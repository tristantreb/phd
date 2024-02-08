"""
Use functions in this file to define the model's variables, their name, discretisation parameters, and priors (if parent) or conditional probability tables (if child)

Each function corresponds to a full set of variables to be plugged into a bayesian network
"""

import src.models.cpts.load as cptloader
import src.models.helpers as mh


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

    HFEV1 = mh.variableNode("Healthy FEV1 (L)", 1, 6, 0.05, prior=hfev1_prior)
    ecFEV1 = mh.variableNode("ecFEV1 (L)", 0, 6, 0.05, prior=None)
    # Lowest predicted FEV1 is 15% (AR = 1-predictedFEV1)
    AR = mh.variableNode("Airway resistance (%)", 0, 90, 2, prior={"type": "uniform"})

    # Res 0.5 takes 19s, res 0.2 takes 21s
    HO2Sat = mh.variableNode(
        "Healthy O2 saturation (%)", 90, 100, 0.5, prior=ho2sat_prior
    )
    # Highest drop is 92% (for AR = 90%)
    # Hence the lowest O2SatFFA is 90 * 0.92 = 82.8
    O2SatFFA = mh.variableNode(
        "O2 saturation if fully functional alveoli (%)", 80, 100, 0.5, prior=None
    )
    # O2 sat can't be below 70%.
    # If there's no airway resistance, it should still be possible to reach 70% O2 sat
    # Hence, min IA is 30% because i
    IA = mh.variableNode("Inactive alveoli (%)", 0, 30, 1, prior={"type": "uniform"})
    # In reality O2 sat can't be below 70%.
    # However, the CPT should account for the fact that the lowest O2 sat is 82.8%.
    # 82.8-30 = 52.8%
    # TODO: should we hardcode the fact that the sum of AR and IA should not be below 70% O2 Sat?
    UO2Sat = mh.variableNode("Underlying O2 saturation (%)", 50, 100, 0.5, prior=None)
    O2Sat = mh.variableNode("O2 saturation (%)", 49.5, 100.5, 1, prior=None)

    # Calculate CPTs
    ecFEV1.prior = cptloader.get_cpt([ecFEV1, HFEV1, AR])
    O2SatFFA.prior = cptloader.get_cpt([O2SatFFA, HO2Sat, AR])
    UO2Sat.prior = cptloader.get_cpt([UO2Sat, O2SatFFA, IA])
    O2Sat.prior = cptloader.get_cpt([O2Sat, UO2Sat])

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

    HFEV1 = mh.variableNode("Healthy FEV1 (L)", 1, 6, 0.05, prior=hfev1_prior)
    ecFEV1 = mh.variableNode("ecFEV1 (L)", 0, 6, 0.05, prior=None)
    # Lowest predicted FEV1 is 15% (AR = 1-predictedFEV1)
    AR = mh.variableNode("Airway resistance (%)", 0, 90, 2, prior=None)

    # Res 0.5 takes 19s, res 0.2 takes 21s
    HO2Sat = mh.variableNode(
        "Healthy O2 saturation (%)", 90, 100, 0.5, prior=ho2sat_prior
    )
    # Highest drop is 92% (for AR = 90%)
    # Hence the lowest O2SatFFA is 90 * 0.92 = 82.8
    O2SatFFA = mh.variableNode(
        "O2 saturation if fully functional alveoli (%)", 80, 100, 0.5, prior=None
    )
    # O2 sat can't be below 70%.
    # If there's no airway resistance, it should still be possible to reach 70% O2 sat
    # Hence, min IA is 30% because i
    IA = mh.variableNode("Inactive alveoli (%)", 0, 30, 1, prior=None)
    # In reality O2 sat can't be below 70%.
    # However, the CPT should account for the fact that the lowest O2 sat is 82.8%.
    # 82.8-30 = 52.8%
    # TODO: should we hardcode the fact that the sum of AR and IA should not be below 70% O2 Sat?
    UO2Sat = mh.variableNode("Underlying O2 saturation (%)", 50, 100, 0.5, prior=None)
    O2Sat = mh.variableNode("O2 saturation (%)", 49.5, 100.5, 1, prior=None)

    # Calculate CPTs
    ecFEV1.prior = cptloader.get_cpt([ecFEV1, HFEV1, AR])
    O2SatFFA.prior = cptloader.get_cpt([O2SatFFA, HO2Sat, AR])
    UO2Sat.prior = cptloader.get_cpt([UO2Sat, O2SatFFA, IA])
    O2Sat.prior = cptloader.get_cpt([O2Sat, UO2Sat])

    AR.prior = ar_prior
    IA.prior = ia_prior

    return HFEV1, ecFEV1, AR, HO2Sat, O2SatFFA, IA, UO2Sat, O2Sat
