"""
Use functions in this file to build the lung health models using the PGMPY library

Each function corresponds to a different bayesian network
"""

from pgmpy.factors.discrete import DiscreteFactor, TabularCPD
from pgmpy.inference import BeliefPropagation
from pgmpy.models import BayesianNetwork, FactorGraph


def fev1_o2sat_point_in_time_model(
    HFEV1, ecFEV1, AR, HO2Sat, O2SatFFA, IA, UO2Sat, O2Sat
):
    """
    AR and IA have no direct link in this model
    """
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
    return model, inf_alg


def fev1_o2sat_point_in_time_model_2(
    HFEV1, ecFEV1, AR, HO2Sat, O2SatFFA, IA, UO2Sat, O2Sat
):
    """
    Update: AR causes IA according to a specific factor
    """
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
            (AR.name, IA.name),
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
    return model, inf_alg


def fev1_o2sat_point_in_time_factor_graph(
    HFEV1, ecFEV1, AR, HO2Sat, O2SatFFA, IA, UO2Sat, O2Sat, check_model=False
):
    """
    AR and IA have no direct link in this model
    """

    phi1 = DiscreteFactor(
        [ecFEV1.name, HFEV1.name, AR.name],
        [len(ecFEV1.bins), len(HFEV1.bins), len(AR.bins)],
        ecFEV1.prior.reshape(len(ecFEV1.bins), len(HFEV1.bins), len(AR.bins)),
    )

    G = FactorGraph()
    G.add_nodes_from([HFEV1.name, ecFEV1.name, AR.name])
    G.add_factors(phi1)
    G.add_edges_from([(HFEV1.name, phi1), (AR.name, phi1), (phi1, ecFEV1.name)])

    if check_model:
        assert G.check_model() == True

    return G
