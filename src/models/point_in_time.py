from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import BeliefPropagation
from pgmpy.models import BayesianNetwork

import src.models.cpts.load as cptloader
import src.models.helpers as mh


def build(hfev1_prior, ho2sat_prior):
    """
    This is a point in time model with:
    FEV1 = HFEV1 * (1-AR)
    O2SatFFA = HO2Sat * drop_func(AR)

    """
    vars = calc_cpts(hfev1_prior, ho2sat_prior)
    model, inf_alg = build_pgmpy_model(*vars)
    return model, inf_alg, *vars


def calc_cpts(hfev1_prior, ho2sat_prior):
    # Build variables
    # Setting resolution of 0.05 to avoud rounding errors for AR
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
    # IA.prior = cptloader.get_cpt([IA, AR])
    UO2Sat.prior = cptloader.get_cpt([UO2Sat, O2SatFFA, IA])
    O2Sat.prior = cptloader.get_cpt([O2Sat, UO2Sat])

    return (HFEV1, ecFEV1, AR, HO2Sat, O2SatFFA, IA, UO2Sat, O2Sat)


def build_pgmpy_model(HFEV1, ecFEV1, AR, HO2Sat, O2SatFFA, IA, UO2Sat, O2Sat):
    prior_hfev1 = TabularCPD(
        variable=HFEV1.name,
        variable_card=len(HFEV1.bins),
        values=HFEV1.prior,
        evidence=[],
        evidence_card=[],
    )
    cpt_ecfev1 = TabularCPD(
        variable=ecFEV1.name,
        variable_card=len(ecFEV1.bins),
        values=ecFEV1.prior,
        evidence=[HFEV1.name, AR.name],
        evidence_card=[len(HFEV1.bins), len(AR.bins)],
    )

    prior_ar = TabularCPD(
        variable=AR.name,
        variable_card=len(AR.bins),
        values=AR.prior,
        evidence=[],
        evidence_card=[],
    )
    prior_ho2sat = TabularCPD(
        variable=HO2Sat.name,
        variable_card=len(HO2Sat.bins),
        values=HO2Sat.prior,
        evidence=[],
        evidence_card=[],
    )
    cpt_o2satffa = TabularCPD(
        variable=O2SatFFA.name,
        variable_card=len(O2SatFFA.bins),
        values=O2SatFFA.prior,
        evidence=[HO2Sat.name, AR.name],
        evidence_card=[len(HO2Sat.bins), len(AR.bins)],
    )
    cpt_ia = TabularCPD(
        variable=IA.name,
        variable_card=len(IA.bins),
        values=IA.prior,
        evidence=[],
        evidence_card=[],
        # evidence=[AR.name],
        # evidence_card=[len(AR.bins)],
    )
    cpt_uo2sat = TabularCPD(
        variable=UO2Sat.name,
        variable_card=len(UO2Sat.bins),
        values=UO2Sat.prior,
        evidence=[O2SatFFA.name, IA.name],
        evidence_card=[len(O2SatFFA.bins), len(IA.bins)],
    )
    cpt_o2sat = TabularCPD(
        variable=O2Sat.name,
        variable_card=len(O2Sat.bins),
        values=O2Sat.prior,
        evidence=[UO2Sat.name],
        evidence_card=[len(UO2Sat.bins)],
    )

    model = BayesianNetwork(
        [
            (HFEV1.name, ecFEV1.name),
            (AR.name, ecFEV1.name),
            (HO2Sat.name, O2SatFFA.name),
            (AR.name, O2SatFFA.name),
            # (AR.name, IA.name),
            (O2SatFFA.name, UO2Sat.name),
            (IA.name, UO2Sat.name),
            (UO2Sat.name, O2Sat.name),
        ]
    )

    model.add_cpds(
        cpt_ecfev1,
        prior_ar,
        prior_hfev1,
        prior_ho2sat,
        cpt_o2satffa,
        cpt_ia,
        cpt_uo2sat,
        cpt_o2sat,
    )

    model.check_model()
    inf_alg = BeliefPropagation(model)
    return (model, inf_alg)
