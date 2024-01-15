from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import BeliefPropagation
from pgmpy.models import BayesianNetwork

import src.modelling_o2.ia as ia
import src.modelling_o2.o2satffa as o2satffa
import src.modelling_o2.unbiased_o2sat as uo2sat
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
        "O2 sat if fully functional alveoli (%)", 80, 100, 0.5, prior=None
    )
    # O2 sat can't be below 70%.
    # If there's no airway resistance, it should still be possible to reach 70% O2 sat
    # Hence, min IA is 30% because i
    IA = mh.variableNode("Inactive alveoli (%)", 0, 30, 1, prior=None)
    # In reality O2 sat can't be below 70%.
    # However, the CPT should account for the fact that the lowest O2 sat is 82.8%.
    # 82.8-30 = 52.8%
    # TODO: should we hardcode the fact that the sum of AR and IA should not be below 70% O2 Sat?
    UO2Sat = mh.variableNode("Unbiased O2 saturation (%)", 50, 100, 0.5, prior=None)

    # Calculate CPTs
    ecFEV1.prior = mh.calc_pgmpy_cpt_X_x_1_minus_Y(HFEV1, AR, ecFEV1)
    O2SatFFA.prior = o2satffa.calc_cpt(O2SatFFA, HO2Sat, AR, debug=False)
    IA.prior = ia.calc_cpt(IA, AR, debug=False)
    UO2Sat.prior = uo2sat.calc_cpt(UO2Sat, O2SatFFA, IA)

    return (HFEV1, ecFEV1, AR, HO2Sat, O2SatFFA, IA, UO2Sat)


def build_pgmpy_model(HFEV1, ecFEV1, AR, HO2Sat, O2SatFFA, IA, UO2Sat):
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
        evidence=[AR.name],
        evidence_card=[len(AR.bins)],
    )
    cpt_uo2sat = TabularCPD(
        variable=UO2Sat.name,
        variable_card=len(UO2Sat.bins),
        values=UO2Sat.prior,
        evidence=[O2SatFFA.name, IA.name],
        evidence_card=[len(O2SatFFA.bins), len(IA.bins)],
    )

    model = BayesianNetwork(
        [
            (HFEV1.name, ecFEV1.name),
            (AR.name, ecFEV1.name),
            (HO2Sat.name, O2SatFFA.name),
            (AR.name, O2SatFFA.name),
            (AR.name, IA.name),
            (O2SatFFA.name, UO2Sat.name),
            (IA.name, UO2Sat.name),
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
    )

    model.check_model()
    inf_alg = BeliefPropagation(model)
    return (model, inf_alg)
