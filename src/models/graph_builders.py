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
        variable_card=HFEV1.card,
        values=HFEV1.cpt.reshape(-1, 1),
        evidence=[],
        evidence_card=[],
    )


def build_pgmpy_ecfev1_cpt(ecFEV1, HFEV1, AR):
    return TabularCPD(
        variable=ecFEV1.name,
        variable_card=ecFEV1.card,
        values=ecFEV1.cpt.reshape(ecFEV1.card, -1),
        evidence=[HFEV1.name, AR.name],
        evidence_card=[HFEV1.card, AR.card],
    )


def build_pgmpy_ar_prior(AR):
    return TabularCPD(
        variable=AR.name,
        variable_card=AR.card,
        values=AR.cpt.reshape(-1, 1),
        evidence=[],
        evidence_card=[],
    )


def build_pgmpy_ho2sat_prior(HO2Sat):
    return TabularCPD(
        variable=HO2Sat.name,
        variable_card=HO2Sat.card,
        values=HO2Sat.cpt.reshape(-1, 1),
        evidence=[],
        evidence_card=[],
    )


def build_pgmpy_o2satffa_cpt(O2SatFFA, HO2Sat, AR):
    return TabularCPD(
        variable=O2SatFFA.name,
        variable_card=O2SatFFA.card,
        values=O2SatFFA.cpt.reshape(O2SatFFA.card, -1),
        evidence=[HO2Sat.name, AR.name],
        evidence_card=[HO2Sat.card, AR.card],
    )


def build_pgmpy_ia_prior(IA):
    return TabularCPD(
        variable=IA.name,
        variable_card=IA.card,
        values=IA.cpt.reshape(-1, 1),
        evidence=[],
        evidence_card=[],
    )


def build_pgmpy_ia_cpt(IA, AR):
    return TabularCPD(
        variable=IA.name,
        variable_card=IA.card,
        values=IA.cpt.reshape(IA.card, -1),
        evidence=[AR.name],
        evidence_card=[AR.card],
    )


def build_pgmpy_uo2sat_cpt(UO2Sat, O2SatFFA, IA):
    return TabularCPD(
        variable=UO2Sat.name,
        variable_card=UO2Sat.card,
        values=UO2Sat.cpt.reshape(UO2Sat.card, -1),
        evidence=[O2SatFFA.name, IA.name],
        evidence_card=[O2SatFFA.card, IA.card],
    )


def build_pgmpy_o2sat_cpt(O2Sat, UO2Sat):
    return TabularCPD(
        variable=O2Sat.name,
        variable_card=O2Sat.card,
        values=O2Sat.cpt,
        evidence=[UO2Sat.name],
        evidence_card=[UO2Sat.card],
    )


def build_pgmpy_ecfef2575prectecfev1_cpt(ecFEF2575prctecFEV1, AR):
    return TabularCPD(
        variable=ecFEF2575prctecFEV1.name,
        variable_card=ecFEF2575prctecFEV1.card,
        values=ecFEF2575prctecFEV1.cpt.reshape(ecFEF2575prctecFEV1.card, -1),
        evidence=[AR.name],
        evidence_card=[AR.card],
    )


def build_pgmpy_hfev1_factor_fn(HFEV1):
    return DiscreteFactor([HFEV1.name], [HFEV1.card], HFEV1.cpt)


def build_pgmpy_ecfev1_factor_fn(ecFEV1, HFEV1, AR):
    return DiscreteFactor(
        [ecFEV1.name, HFEV1.name, AR.name],
        [ecFEV1.card, HFEV1.card, AR.card],
        ecFEV1.cpt,
    )


def build_pgmpy_ecfef2575prctecfev1_factor_fn(ecFEF2575prctecFEV1, AR):
    return DiscreteFactor(
        [ecFEF2575prctecFEV1.name, AR.name],
        [ecFEF2575prctecFEV1.card, AR.card],
        ecFEF2575prctecFEV1.cpt,
    )


def build_pgmpy_ar_factor_fn(AR):
    return DiscreteFactor([AR.name], [AR.card], AR.cpt)


def build_pgmpy_ho2sat_factor_fn(HO2Sat):
    return DiscreteFactor([HO2Sat.name], [HO2Sat.card], HO2Sat.cpt)


def build_pgmpy_o2satffa_factor_fn(O2SatFFA, HO2Sat, AR):
    return DiscreteFactor(
        [O2SatFFA.name, HO2Sat.name, AR.name],
        [O2SatFFA.card, HO2Sat.card, AR.card],
        O2SatFFA.cpt,
    )


def build_pgmpy_ia_factor_fn(IA):
    return DiscreteFactor([IA.name], [IA.card], IA.cpt)


def build_pgmpy_uo2sat_factor_fn(UO2Sat, O2SatFFA, IA):
    return DiscreteFactor(
        [UO2Sat.name, O2SatFFA.name, IA.name],
        [UO2Sat.card, O2SatFFA.card, IA.card],
        UO2Sat.cpt,
    )


def build_pgmpy_o2sat_factor_fn(O2Sat, UO2Sat):
    return DiscreteFactor(
        [O2Sat.name, UO2Sat.name], [O2Sat.card, UO2Sat.card], O2Sat.cpt
    )


def fev1_point_in_time_model(HFEV1, ecFEV1, AR):
    prior_hfev1 = build_pgmpy_hfev1_prior(HFEV1)
    cpt_ecfev1 = build_pgmpy_ecfev1_cpt(ecFEV1, HFEV1, AR)
    prior_ar = build_pgmpy_ar_prior(AR)

    model = BayesianNetwork(
        [
            (HFEV1.name, ecFEV1.name),
            (AR.name, ecFEV1.name),
        ]
    )

    model.add_cpds(
        cpt_ecfev1,
        prior_ar,
        prior_hfev1,
    )

    model.check_model()
    return model


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
    prior_hfev1 = build_pgmpy_hfev1_prior(HFEV1)
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


def fev1_o2sat_fef2575_point_in_time_model(
    HFEV1,
    ecFEV1,
    AR,
    HO2Sat,
    O2SatFFA,
    IA,
    UO2Sat,
    O2Sat,
    ecFEF2575prctecFEV1,
    check_model=True,
):
    """
    Update: AR causes IA according to a specific factor
    """
    prior_hfev1 = build_pgmpy_hfev1_prior(HFEV1)
    cpt_ecfev1 = build_pgmpy_ecfev1_cpt(ecFEV1, HFEV1, AR)
    prior_ar = build_pgmpy_ar_prior(AR)
    prior_ho2sat = build_pgmpy_ho2sat_prior(HO2Sat)
    cpt_o2satffa = build_pgmpy_o2satffa_cpt(O2SatFFA, HO2Sat, AR)
    cpt_ia = build_pgmpy_ia_cpt(IA, AR)
    cpt_uo2sat = build_pgmpy_uo2sat_cpt(UO2Sat, O2SatFFA, IA)
    cpt_o2sat = build_pgmpy_o2sat_cpt(O2Sat, UO2Sat)
    cpt_ecFEF2575prctecFEV1 = build_pgmpy_ecfef2575prectecfev1_cpt(
        ecFEF2575prctecFEV1, AR
    )

    model = BayesianNetwork(
        [
            (HFEV1.name, ecFEV1.name),
            (AR.name, ecFEV1.name),
            (HO2Sat.name, O2SatFFA.name),
            (AR.name, O2SatFFA.name),
            (AR.name, IA.name),
            (AR.name, ecFEF2575prctecFEV1.name),
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
        cpt_ecFEF2575prctecFEV1,
    )
    if check_model:
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


def fev1_fef2575_o2sat_point_in_time_factor_graph(
    HFEV1,
    ecFEV1,
    AR,
    HO2Sat,
    O2SatFFA,
    IA,
    UO2Sat,
    O2Sat,
    ecFEF2575prctecFEV1,
    check_model=True,
):
    """
    AR and IA have no direct link in this model
    """

    prior_hfev1 = build_pgmpy_hfev1_factor_fn(HFEV1)
    cpt_ecfev1 = build_pgmpy_ecfev1_factor_fn(ecFEV1, HFEV1, AR)
    cpt_ecFEF2575prctecFEV1 = build_pgmpy_ecfef2575prctecfev1_factor_fn(
        ecFEF2575prctecFEV1, AR
    )
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
            ecFEF2575prctecFEV1.name,
        ]
    )
    G.add_factors(
        prior_hfev1,
        cpt_ecfev1,
        cpt_ecFEF2575prctecFEV1,
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
            (AR.name, cpt_ecFEF2575prctecFEV1),
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
            (cpt_ecFEF2575prctecFEV1, ecFEF2575prctecFEV1.name),
        ]
    )

    if check_model:
        assert G.check_model() == True
        print("Model is valid")

    return G
