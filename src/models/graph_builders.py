"""
Use functions in this file to build the lung health models using the PGMPY library

Each function corresponds to a different bayesian network

Note on PGMPY:
- TabularCPD only accepts 2D arrays. Hence, non 2D CPTs/priors are reshaped to 2D arrays
"""

from pgmpy.factors.discrete import DiscreteFactor, TabularCPD
from pgmpy.models import BayesianNetwork, FactorGraph


def build_pgmpy_hfev1_prior(HFEV1):
    return TabularCPD(
        variable=HFEV1.name,
        variable_card=len(HFEV1.bins),
        values=HFEV1.cpt.reshape(-1, 1),
        evidence=[],
        evidence_card=[],
    )


def build_pgmpy_ecfev1_cpt(ecFEV1, HFEV1, AR):
    return TabularCPD(
        variable=ecFEV1.name,
        variable_card=len(ecFEV1.bins),
        values=ecFEV1.cpt.reshape(len(ecFEV1.bins), -1),
        evidence=[HFEV1.name, AR.name],
        evidence_card=[len(HFEV1.bins), len(AR.bins)],
    )


def build_pgmpy_ar_prior(AR):
    return TabularCPD(
        variable=AR.name,
        variable_card=len(AR.bins),
        values=AR.cpt.reshape(-1, 1),
        evidence=[],
        evidence_card=[],
    )


def build_pgmpy_ho2sat_prior(HO2Sat):
    return TabularCPD(
        variable=HO2Sat.name,
        variable_card=len(HO2Sat.bins),
        values=HO2Sat.cpt.reshape(-1, 1),
        evidence=[],
        evidence_card=[],
    )


def build_pgmpy_o2satffa_cpt(O2SatFFA, HO2Sat, AR):
    return TabularCPD(
        variable=O2SatFFA.name,
        variable_card=len(O2SatFFA.bins),
        values=O2SatFFA.cpt.reshape(len(O2SatFFA.bins), -1),
        evidence=[HO2Sat.name, AR.name],
        evidence_card=[len(HO2Sat.bins), len(AR.bins)],
    )


def build_pgmpy_ia_prior(IA):
    return TabularCPD(
        variable=IA.name,
        variable_card=len(IA.bins),
        values=IA.cpt.reshape(-1, 1),
        evidence=[],
        evidence_card=[],
    )


def build_pgmpy_ia_cpt(IA, AR):
    return TabularCPD(
        variable=IA.name,
        variable_card=len(IA.bins),
        values=IA.cpt,
        evidence=[AR.name],
        evidence_card=[len(AR.bins)],
    )


def build_pgmpy_uo2sat_cpt(UO2Sat, O2SatFFA, IA):
    return TabularCPD(
        variable=UO2Sat.name,
        variable_card=len(UO2Sat.bins),
        values=UO2Sat.cpt.reshape(len(UO2Sat.bins), -1),
        evidence=[O2SatFFA.name, IA.name],
        evidence_card=[len(O2SatFFA.bins), len(IA.bins)],
    )


def build_pgmpy_o2sat_cpt(O2Sat, UO2Sat):
    return TabularCPD(
        variable=O2Sat.name,
        variable_card=len(O2Sat.bins),
        values=O2Sat.cpt,
        evidence=[UO2Sat.name],
        evidence_card=[len(UO2Sat.bins)],
    )


def build_pgmpy_hfev1_factor_fn(HFEV1):
    return DiscreteFactor([HFEV1.name], [len(HFEV1.bins)], HFEV1.cpt)


def build_pgmpy_ecfev1_factor_fn(ecFEV1, HFEV1, AR):
    return DiscreteFactor(
        [ecFEV1.name, HFEV1.name, AR.name],
        [len(ecFEV1.bins), len(HFEV1.bins), len(AR.bins)],
        ecFEV1.cpt,
    )


def build_pgmpy_ar_factor_fn(AR):
    return DiscreteFactor([AR.name], [len(AR.bins)], AR.cpt)


def build_pgmpy_ho2sat_factor_fn(HO2Sat):
    return DiscreteFactor([HO2Sat.name], [len(HO2Sat.bins)], HO2Sat.cpt)


def build_pgmpy_o2satffa_factor_fn(O2SatFFA, HO2Sat, AR):
    return DiscreteFactor(
        [O2SatFFA.name, HO2Sat.name, AR.name],
        [len(O2SatFFA.bins), len(HO2Sat.bins), len(AR.bins)],
        O2SatFFA.cpt,
    )


def build_pgmpy_ia_factor_fn(IA):
    return DiscreteFactor([IA.name], [len(IA.bins)], IA.cpt)


def build_pgmpy_uo2sat_factor_fn(UO2Sat, O2SatFFA, IA):
    return DiscreteFactor(
        [UO2Sat.name, O2SatFFA.name, IA.name],
        [len(UO2Sat.bins), len(O2SatFFA.bins), len(IA.bins)],
        UO2Sat.cpt,
    )


def build_pgmpy_o2sat_factor_fn(O2Sat, UO2Sat):
    return DiscreteFactor(
        [O2Sat.name, UO2Sat.name], [len(O2Sat.bins), len(UO2Sat.bins)], O2Sat.cpt
    )


def fev1_o2sat_point_in_time_model(
    HFEV1, ecFEV1, AR, HO2Sat, O2SatFFA, IA, UO2Sat, O2Sat
):
    """
    AR and IA have no direct link in this model
    """
    prior_hfev1 = build_pgmpy_hfev1_prior(HFEV1)
    cpt_ecfev1 = build_pgmpy_ecfev1_cpt(ecFEV1, HFEV1, AR)
    prior_ar = build_pgmpy_ar_prior(AR)
    prior_ho2sat = build_pgmpy_ho2sat_prior(HO2Sat)
    cpt_o2satffa = build_pgmpy_o2satffa_cpt(O2SatFFA, HO2Sat, AR)
    prior_ia = build_pgmpy_ia_prior(IA)
    cpt_uo2sat = build_pgmpy_uo2sat_cpt(UO2Sat, O2SatFFA, IA)
    cpt_o2sat = build_pgmpy_o2sat_cpt(O2Sat, UO2Sat)

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
        prior_ia,
        cpt_uo2sat,
        cpt_o2sat,
    )

    model.check_model()
    return model


def fev1_o2sat_point_in_time_model_2(
    HFEV1, ecFEV1, AR, HO2Sat, O2SatFFA, IA, UO2Sat, O2Sat
):
    """
    Update: AR causes IA according to a specific factor
    """
    prior_hfev1 = build_pgmpy_hfev1_prior
    cpt_ecfev1 = build_pgmpy_ecfev1_cpt(ecFEV1, HFEV1, AR)
    prior_ar = build_pgmpy_ar_prior(AR)
    prior_ho2sat = build_pgmpy_ho2sat_prior(HO2Sat)
    cpt_o2satffa = build_pgmpy_o2satffa_cpt(O2SatFFA, HO2Sat, AR)
    cpt_ia = build_pgmpy_ia_cpt(IA, AR)
    cpt_uo2sat = build_pgmpy_uo2sat_cpt(UO2Sat, O2SatFFA, IA)
    cpt_o2sat = build_pgmpy_o2sat_cpt(O2Sat, UO2Sat)

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
    return model


def fev1_o2sat_point_in_time_factor_graph(
    HFEV1, ecFEV1, AR, HO2Sat, O2SatFFA, IA, UO2Sat, O2Sat, check_model=True
):
    """
    AR and IA have no direct link in this model
    """

    prior_hfev1 = build_pgmpy_hfev1_factor_fn(HFEV1)
    cpt_ecfev1 = build_pgmpy_ecfev1_factor_fn(ecFEV1, HFEV1, AR)
    prior_ar = build_pgmpy_ar_factor_fn(AR)
    prior_ho2sat = build_pgmpy_ho2sat_factor_fn(HO2Sat)
    cpt_o2satffa = build_pgmpy_o2satffa_factor_fn(O2SatFFA, HO2Sat, AR)
    prior_ia = build_pgmpy_ia_factor_fn(IA)
    cpt_uo2sat = build_pgmpy_uo2sat_factor_fn(UO2Sat, O2SatFFA, IA)
    cpt_o2sat = build_pgmpy_o2sat_factor_fn(O2Sat, UO2Sat)

    G = FactorGraph()
    G.add_nodes_from(
        [
            HFEV1.name,
            ecFEV1.name,
            AR.name,
            HO2Sat.name,
            O2SatFFA.name,
            IA.name,
            UO2Sat.name,
            O2Sat.name,
        ]
    )
    G.add_factors(
        prior_hfev1,
        cpt_ecfev1,
        prior_ar,
        prior_ho2sat,
        cpt_o2satffa,
        prior_ia,
        cpt_uo2sat,
        cpt_o2sat,
    )
    G.add_edges_from(
        [
            (prior_hfev1, HFEV1.name),
            (HFEV1.name, cpt_ecfev1),
            (prior_ar, AR.name),
            (AR.name, cpt_ecfev1),
            (AR.name, cpt_o2satffa),
            (prior_ho2sat, HO2Sat.name),
            (HO2Sat.name, cpt_o2satffa),
            (cpt_o2satffa, O2SatFFA.name),
            (O2SatFFA.name, cpt_uo2sat),
            (prior_ia, IA.name),
            (IA.name, cpt_uo2sat),
            (cpt_uo2sat, UO2Sat.name),
            (UO2Sat.name, cpt_o2sat),
            (cpt_ecfev1, ecFEV1.name),
        ]
    )

    if check_model:
        assert G.check_model() == True
        print("Model is valid")

    return G
