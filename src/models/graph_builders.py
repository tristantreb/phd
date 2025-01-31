"""
Use functions in this file to build the lung health models using the PGMPY library

Each function corresponds to a different bayesian network

Note on PGMPY:
- TabularCPD only accepts 2D arrays. Hence, non 2D CPTs/priors are reshaped to 2D arrays
"""

from copy import deepcopy

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


def build_pgmpy_ecfev1_noise_cpt(ecFEV1, uecFEV1):
    return TabularCPD(
        variable=ecFEV1.name,
        variable_card=ecFEV1.card,
        values=ecFEV1.cpt.reshape(ecFEV1.card, -1),
        evidence=[uecFEV1.name],
        evidence_card=[uecFEV1.card],
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


def build_pgmpy_ecfef2575prctecfev1_cpt(ecFEF2575prctecFEV1, AR):
    return TabularCPD(
        variable=ecFEF2575prctecFEV1.name,
        variable_card=ecFEF2575prctecFEV1.card,
        values=ecFEF2575prctecFEV1.cpt.reshape(ecFEF2575prctecFEV1.card, -1),
        evidence=[AR.name],
        evidence_card=[AR.card],
    )


def build_pgmpy_ar_change_var_prior(S):
    return TabularCPD(
        variable=S.name,
        variable_card=S.card,
        values=S.cpt.reshape(-1, 1),
        evidence=[],
        evidence_card=[],
    )


def build_pgmpy_hfev1_factor_fn(HFEV1):
    return DiscreteFactor([HFEV1.name], [HFEV1.card], HFEV1.cpt)


def build_pgmpy_ecfev1_factor_fn(uecFEV1, HFEV1, AR):
    return DiscreteFactor(
        [uecFEV1.name, HFEV1.name, AR.name],
        [uecFEV1.card, HFEV1.card, AR.card],
        uecFEV1.cpt,
    )


def build_pgmpy_ecfev1_noise_factor_fn(ecFEV1, uecFEV1):
    return DiscreteFactor(
        [ecFEV1.name, uecFEV1.name],
        [ecFEV1.card, uecFEV1.card],
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
    cpt_ecFEF2575prctecFEV1 = build_pgmpy_ecfef2575prctecfev1_cpt(
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


def fev1_o2sat_two_days_model(
    HFEV1,
    ecFEV1,
    AR,
    HO2Sat,
    O2SatFFA,
    IA,
    UO2Sat,
    O2Sat,
    check_model=True,
):
    """
    No direct link between AR and IA
    """

    ecFEV1_2 = deepcopy(ecFEV1)
    ecFEV1_2.name = f"{ecFEV1.name} day 2"
    AR_2 = deepcopy(AR)
    AR_2.name = f"{AR.name} day 2"
    O2SatFFA_2 = deepcopy(O2SatFFA)
    O2SatFFA_2.name = f"{O2SatFFA.name} day 2"
    IA_2 = deepcopy(IA)
    IA_2.name = f"{IA.name} day 2"
    UO2Sat_2 = deepcopy(UO2Sat)
    UO2Sat_2.name = f"{UO2Sat.name} day 2"
    O2Sat_2 = deepcopy(O2Sat)
    O2Sat_2.name = f"{O2Sat.name} day 2"

    # Shared priors
    prior_hfev1 = build_pgmpy_hfev1_prior(HFEV1)
    prior_ho2sat = build_pgmpy_ho2sat_prior(HO2Sat)
    # Day 1
    cpt_ecfev1 = build_pgmpy_ecfev1_cpt(ecFEV1, HFEV1, AR)
    prior_ar = build_pgmpy_ar_prior(AR)
    cpt_o2satffa = build_pgmpy_o2satffa_cpt(O2SatFFA, HO2Sat, AR)
    prior_ia = build_pgmpy_ia_prior(IA)
    cpt_uo2sat = build_pgmpy_uo2sat_cpt(UO2Sat, O2SatFFA, IA)
    cpt_o2sat = build_pgmpy_o2sat_cpt(O2Sat, UO2Sat)
    # Day 2
    cpt_ecfev1_2 = build_pgmpy_ecfev1_cpt(ecFEV1_2, HFEV1, AR_2)
    prior_ar_2 = build_pgmpy_ar_prior(AR_2)
    cpt_o2satffa_2 = build_pgmpy_o2satffa_cpt(O2SatFFA_2, HO2Sat, AR_2)
    prior_ia_2 = build_pgmpy_ia_prior(IA_2)
    cpt_uo2sat_2 = build_pgmpy_uo2sat_cpt(UO2Sat_2, O2SatFFA_2, IA_2)
    cpt_o2sat_2 = build_pgmpy_o2sat_cpt(O2Sat_2, UO2Sat_2)

    # HFEV1 and HO2Sat are shared between day 1 and day2
    model = BayesianNetwork(
        [
            # Day 1
            (HFEV1.name, ecFEV1.name),
            (HO2Sat.name, O2SatFFA.name),
            (AR.name, ecFEV1.name),
            (AR.name, O2SatFFA.name),
            (O2SatFFA.name, UO2Sat.name),
            (IA.name, UO2Sat.name),
            (UO2Sat.name, O2Sat.name),
            # Day 2
            (HFEV1.name, ecFEV1_2.name),
            (AR_2.name, ecFEV1_2.name),
            (HO2Sat.name, O2SatFFA_2.name),
            (AR_2.name, O2SatFFA_2.name),
            (O2SatFFA_2.name, UO2Sat_2.name),
            (IA_2.name, UO2Sat_2.name),
            (UO2Sat_2.name, O2Sat_2.name),
        ]
    )

    model.add_cpds(
        # Shared
        prior_hfev1,
        prior_ho2sat,
        # Day 1
        cpt_ecfev1,
        prior_ar,
        cpt_o2satffa,
        prior_ia,
        cpt_uo2sat,
        cpt_o2sat,
        # Day 2
        cpt_ecfev1_2,
        prior_ar_2,
        cpt_o2satffa_2,
        prior_ia_2,
        cpt_uo2sat_2,
        cpt_o2sat_2,
    )
    if check_model:
        model.check_model()
    return (
        model,
        HFEV1,
        HO2Sat,
        ecFEV1,
        AR,
        O2SatFFA,
        IA,
        UO2Sat,
        O2Sat,
        ecFEV1_2,
        AR_2,
        O2SatFFA_2,
        IA_2,
        UO2Sat_2,
        O2Sat_2,
    )


def fev1_o2sat_n_days_model(
    n,
    HFEV1,
    ecFEV1,
    AR,
    HO2Sat,
    O2SatFFA,
    IA,
    UO2Sat,
    O2Sat,
    check_model=True,
):
    """
    No direct link between AR and IA
    """

    def create_var_for_day(var, day):
        var_ = deepcopy(var)
        var_.name = f"{var.name} day {day}"
        return var_

    AR_vars = [create_var_for_day(AR, i) for i in range(1, n + 1)]
    ecFEV1_vars = [create_var_for_day(ecFEV1, i) for i in range(1, n + 1)]
    O2SatFFA_vars = [create_var_for_day(O2SatFFA, i) for i in range(1, n + 1)]
    IA_vars = [create_var_for_day(IA, i) for i in range(1, n + 1)]
    UO2Sat_vars = [create_var_for_day(UO2Sat, i) for i in range(1, n + 1)]
    O2Sat_vars = [create_var_for_day(O2Sat, i) for i in range(1, n + 1)]

    # Priors and CPTs
    AR_priors = [build_pgmpy_ar_prior(AR_) for AR_ in AR_vars]
    ecFEV1_cpts = [
        build_pgmpy_ecfev1_cpt(ecFEV1_, HFEV1, AR_)
        for ecFEV1_, AR_ in zip(ecFEV1_vars, AR_vars)
    ]
    O2SatFFA_cpts = [
        build_pgmpy_o2satffa_cpt(O2SatFFA_, HO2Sat, AR_)
        for O2SatFFA_, AR_ in zip(O2SatFFA_vars, AR_vars)
    ]
    IA_priors = [build_pgmpy_ia_prior(IA_) for IA_ in IA_vars]
    UO2Sat_cpts = [
        build_pgmpy_uo2sat_cpt(UO2Sat_, O2SatFFA_, IA_)
        for UO2Sat_, O2SatFFA_, IA_ in zip(UO2Sat_vars, O2SatFFA_vars, IA_vars)
    ]
    O2Sat_cpts = [
        build_pgmpy_o2sat_cpt(O2Sat_, UO2Sat_)
        for O2Sat_, UO2Sat_ in zip(O2Sat_vars, UO2Sat_vars)
    ]

    # Shared priors
    prior_hfev1 = build_pgmpy_hfev1_prior(HFEV1)
    prior_ho2sat = build_pgmpy_ho2sat_prior(HO2Sat)

    # Make sure the days are at the same location in all variable lists
    for i in range(1, n + 1):
        assert AR_vars[i - 1].name == f"{AR.name} day {i}"
        assert ecFEV1_vars[i - 1].name == f"{ecFEV1.name} day {i}"
        assert O2SatFFA_vars[i - 1].name == f"{O2SatFFA.name} day {i}"
        assert IA_vars[i - 1].name == f"{IA.name} day {i}"
        assert UO2Sat_vars[i - 1].name == f"{UO2Sat.name} day {i}"
        assert O2Sat_vars[i - 1].name == f"{O2Sat.name} day {i}"

    network = []
    # HFEV1 and HO2Sat are shared between days
    for i in range(0, n):
        network += [
            (HFEV1.name, ecFEV1_vars[i].name),
            (HO2Sat.name, O2SatFFA_vars[i].name),
            (AR_vars[i].name, ecFEV1_vars[i].name),
            (AR_vars[i].name, O2SatFFA_vars[i].name),
            (O2SatFFA_vars[i].name, UO2Sat_vars[i].name),
            (IA_vars[i].name, UO2Sat_vars[i].name),
            (UO2Sat_vars[i].name, O2Sat_vars[i].name),
        ]

    model = BayesianNetwork(network)

    model.add_cpds(
        # Shared
        prior_hfev1,
        prior_ho2sat,
        # Days priors and CPTs
        *AR_priors,
        *ecFEV1_cpts,
        *O2SatFFA_cpts,
        *IA_priors,
        *UO2Sat_cpts,
        *O2Sat_cpts,
    )

    if check_model:
        model.check_model()
    return (
        model,
        HFEV1,
        HO2Sat,
        AR_vars,
        ecFEV1_vars,
        O2SatFFA_vars,
        IA_vars,
        UO2Sat_vars,
        O2Sat_vars,
    )


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
            (cpt_o2sat, O2Sat.name),
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
            (cpt_o2sat, O2Sat.name),
            (cpt_ecfev1, ecFEV1.name),
            (cpt_ecFEF2575prctecFEV1, ecFEF2575prctecFEV1.name),
        ]
    )

    if check_model:
        assert G.check_model() == True
        print("Model is valid")

    return G


def fev1_fef2575_o2sat_point_in_time_noise_factor_graph(
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
    check_model=True,
):
    """
    AR and IA have no direct link in this model
    """

    prior_hfev1 = build_pgmpy_hfev1_factor_fn(HFEV1)
    cpt_uecfev1 = build_pgmpy_ecfev1_factor_fn(uecFEV1, HFEV1, AR)
    cpt_ecfev1 = build_pgmpy_ecfev1_noise_factor_fn(ecFEV1, uecFEV1)
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
            uecFEV1.name,
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
        cpt_uecfev1,
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
            (HFEV1.name, cpt_uecfev1),
            (prior_ar, AR.name),
            (AR.name, cpt_uecfev1),
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
            (cpt_o2sat, O2Sat.name),
            (cpt_uecfev1, uecFEV1.name),
            (uecFEV1.name, cpt_ecfev1),
            (cpt_ecfev1, ecFEV1.name),
            (cpt_ecFEF2575prctecFEV1, ecFEF2575prctecFEV1.name),
        ]
    )

    if check_model:
        assert G.check_model() == True
        print("Model is valid")

    return G


def fev1_o2sat_fef2575_noise_n_days_model(
    n,
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
    check_model=True,
):
    """
    No direct link between AR and IA
    """

    def create_var_for_day(var, day):
        var_ = deepcopy(var)
        var_.name = f"{var.name} day {day}"
        return var_

    AR_vars = [create_var_for_day(AR, i) for i in range(1, n + 1)]
    uecFEV1_vars = [create_var_for_day(uecFEV1, i) for i in range(1, n + 1)]
    ecFEV1_vars = [create_var_for_day(ecFEV1, i) for i in range(1, n + 1)]
    ecFEF2575prctecFEV1_vars = [
        create_var_for_day(ecFEF2575prctecFEV1, i) for i in range(1, n + 1)
    ]
    O2SatFFA_vars = [create_var_for_day(O2SatFFA, i) for i in range(1, n + 1)]
    IA_vars = [create_var_for_day(IA, i) for i in range(1, n + 1)]
    UO2Sat_vars = [create_var_for_day(UO2Sat, i) for i in range(1, n + 1)]
    O2Sat_vars = [create_var_for_day(O2Sat, i) for i in range(1, n + 1)]

    # Priors and CPTs
    AR_priors = [build_pgmpy_ar_prior(AR_) for AR_ in AR_vars]

    uecFEV1_cpts = [
        build_pgmpy_ecfev1_cpt(uecFEV1_, HFEV1, AR_)
        for uecFEV1_, AR_ in zip(uecFEV1_vars, AR_vars)
    ]
    ecFEV1_cpts = [
        build_pgmpy_ecfev1_noise_cpt(ecFEV1_, uecFEV1_)
        for ecFEV1_, uecFEV1_ in zip(ecFEV1_vars, uecFEV1_vars)
    ]
    ecFEF2575prctecFEV1_cpts = [
        build_pgmpy_ecfef2575prctecfev1_cpt(
            ecFEF2575prctecFEV1_,
            AR_,
        )
        for ecFEF2575prctecFEV1_, AR_ in zip(ecFEF2575prctecFEV1_vars, AR_vars)
    ]
    O2SatFFA_cpts = [
        build_pgmpy_o2satffa_cpt(O2SatFFA_, HO2Sat, AR_)
        for O2SatFFA_, AR_ in zip(O2SatFFA_vars, AR_vars)
    ]
    IA_priors = [build_pgmpy_ia_prior(IA_) for IA_ in IA_vars]
    UO2Sat_cpts = [
        build_pgmpy_uo2sat_cpt(UO2Sat_, O2SatFFA_, IA_)
        for UO2Sat_, O2SatFFA_, IA_ in zip(UO2Sat_vars, O2SatFFA_vars, IA_vars)
    ]
    O2Sat_cpts = [
        build_pgmpy_o2sat_cpt(O2Sat_, UO2Sat_)
        for O2Sat_, UO2Sat_ in zip(O2Sat_vars, UO2Sat_vars)
    ]

    # Shared priors
    prior_hfev1 = build_pgmpy_hfev1_prior(HFEV1)
    prior_ho2sat = build_pgmpy_ho2sat_prior(HO2Sat)

    # Make sure the days are at the same location in all variable lists
    for i in range(1, n + 1):
        assert AR_vars[i - 1].name == f"{AR.name} day {i}"
        assert uecFEV1_vars[i - 1].name == f"{uecFEV1.name} day {i}"
        assert ecFEV1_vars[i - 1].name == f"{ecFEV1.name} day {i}"
        assert (
            ecFEF2575prctecFEV1_vars[i - 1].name
            == f"{ecFEF2575prctecFEV1.name} day {i}"
        )
        assert O2SatFFA_vars[i - 1].name == f"{O2SatFFA.name} day {i}"
        assert IA_vars[i - 1].name == f"{IA.name} day {i}"
        assert UO2Sat_vars[i - 1].name == f"{UO2Sat.name} day {i}"
        assert O2Sat_vars[i - 1].name == f"{O2Sat.name} day {i}"

    network = []
    # HFEV1 and HO2Sat are shared between days
    for i in range(0, n):
        network += [
            (HFEV1.name, uecFEV1_vars[i].name),
            (HO2Sat.name, O2SatFFA_vars[i].name),
            (AR_vars[i].name, uecFEV1_vars[i].name),
            (uecFEV1_vars[i].name, ecFEV1_vars[i].name),
            (AR_vars[i].name, ecFEF2575prctecFEV1_vars[i].name),
            (AR_vars[i].name, O2SatFFA_vars[i].name),
            (O2SatFFA_vars[i].name, UO2Sat_vars[i].name),
            (IA_vars[i].name, UO2Sat_vars[i].name),
            (UO2Sat_vars[i].name, O2Sat_vars[i].name),
        ]

    model = BayesianNetwork(network)

    model.add_cpds(
        # Shared
        prior_hfev1,
        prior_ho2sat,
        # Days priors and CPTs
        *AR_priors,
        *uecFEV1_cpts,
        *ecFEV1_cpts,
        *ecFEF2575prctecFEV1_cpts,
        *O2SatFFA_cpts,
        *IA_priors,
        *UO2Sat_cpts,
        *O2Sat_cpts,
    )

    if check_model:
        model.check_model()
    return (
        model,
        HFEV1,
        HO2Sat,
        AR_vars,
        uecFEV1_vars,
        ecFEV1_vars,
        O2SatFFA_vars,
        IA_vars,
        UO2Sat_vars,
        O2Sat_vars,
        ecFEF2575prctecFEV1_vars,
    )


def fev1_o2sat_fef2575_noise_n_days_model_temporal_ar(
    n,
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
    ar_change_cpt_state=-1,
    check_model=True,
):
    """
    No direct link between AR and IA

    The AR change factor is fixed to the cpt[:, :, ar_change_cpt_state] for all days
    """

    def create_var_for_day(var, day):
        var_ = deepcopy(var)
        var_.name = f"{var.name} day {day}"
        return var_

    AR_vars = [create_var_for_day(AR, i) for i in range(1, n + 1)]
    uecFEV1_vars = [create_var_for_day(uecFEV1, i) for i in range(1, n + 1)]
    ecFEV1_vars = [create_var_for_day(ecFEV1, i) for i in range(1, n + 1)]
    ecFEF2575prctecFEV1_vars = [
        create_var_for_day(ecFEF2575prctecFEV1, i) for i in range(1, n + 1)
    ]
    O2SatFFA_vars = [create_var_for_day(O2SatFFA, i) for i in range(1, n + 1)]
    IA_vars = [create_var_for_day(IA, i) for i in range(1, n + 1)]
    UO2Sat_vars = [create_var_for_day(UO2Sat, i) for i in range(1, n + 1)]
    O2Sat_vars = [create_var_for_day(O2Sat, i) for i in range(1, n + 1)]

    # Priors and CPTs
    # AR first day prior shall be uniform
    AR_priors = []
    AR_priors.append(
        TabularCPD(
            variable=AR_vars[0].name,
            variable_card=AR_vars[0].card,
            values=AR_vars[0].first_day_prior.reshape(AR_vars[0].card, -1),
            evidence=[],
            evidence_card=[],
        )
    )
    # For other days the cpt depends the number of days elapsed
    # For testing, n days elapsed = 1
    for i in range(1, n):
        print(
            f"{AR_vars[i].name}, cpt: {AR_vars[i].change_cpt[:, :, ar_change_cpt_state].shape}"
        )
        AR_priors.append(
            TabularCPD(
                variable=AR_vars[i].name,
                variable_card=AR_vars[i].card,
                values=AR_vars[i].change_cpt[:, :, ar_change_cpt_state],
                evidence=[AR_vars[i - 1].name],
                evidence_card=[AR_vars[i - 1].card],
            )
        )

    uecFEV1_cpts = [
        build_pgmpy_ecfev1_cpt(uecFEV1_, HFEV1, AR_)
        for uecFEV1_, AR_ in zip(uecFEV1_vars, AR_vars)
    ]
    ecFEV1_cpts = [
        build_pgmpy_ecfev1_noise_cpt(ecFEV1_, uecFEV1_)
        for ecFEV1_, uecFEV1_ in zip(ecFEV1_vars, uecFEV1_vars)
    ]
    ecFEF2575prctecFEV1_cpts = [
        build_pgmpy_ecfef2575prctecfev1_cpt(
            ecFEF2575prctecFEV1_,
            AR_,
        )
        for ecFEF2575prctecFEV1_, AR_ in zip(ecFEF2575prctecFEV1_vars, AR_vars)
    ]
    O2SatFFA_cpts = [
        build_pgmpy_o2satffa_cpt(O2SatFFA_, HO2Sat, AR_)
        for O2SatFFA_, AR_ in zip(O2SatFFA_vars, AR_vars)
    ]
    IA_priors = [build_pgmpy_ia_prior(IA_) for IA_ in IA_vars]
    UO2Sat_cpts = [
        build_pgmpy_uo2sat_cpt(UO2Sat_, O2SatFFA_, IA_)
        for UO2Sat_, O2SatFFA_, IA_ in zip(UO2Sat_vars, O2SatFFA_vars, IA_vars)
    ]
    O2Sat_cpts = [
        build_pgmpy_o2sat_cpt(O2Sat_, UO2Sat_)
        for O2Sat_, UO2Sat_ in zip(O2Sat_vars, UO2Sat_vars)
    ]

    # Shared priors
    prior_hfev1 = build_pgmpy_hfev1_prior(HFEV1)
    prior_ho2sat = build_pgmpy_ho2sat_prior(HO2Sat)

    # Make sure the days are at the same location in all variable lists
    for i in range(1, n + 1):
        assert AR_vars[i - 1].name == f"{AR.name} day {i}"
        assert uecFEV1_vars[i - 1].name == f"{uecFEV1.name} day {i}"
        assert ecFEV1_vars[i - 1].name == f"{ecFEV1.name} day {i}"
        assert (
            ecFEF2575prctecFEV1_vars[i - 1].name
            == f"{ecFEF2575prctecFEV1.name} day {i}"
        )
        assert O2SatFFA_vars[i - 1].name == f"{O2SatFFA.name} day {i}"
        assert IA_vars[i - 1].name == f"{IA.name} day {i}"
        assert UO2Sat_vars[i - 1].name == f"{UO2Sat.name} day {i}"
        assert O2Sat_vars[i - 1].name == f"{O2Sat.name} day {i}"

    network = []
    # HFEV1 and HO2Sat are shared between days
    for i in range(0, n):
        network += [
            (HFEV1.name, uecFEV1_vars[i].name),
            (HO2Sat.name, O2SatFFA_vars[i].name),
            (AR_vars[i].name, uecFEV1_vars[i].name),
            (uecFEV1_vars[i].name, ecFEV1_vars[i].name),
            (AR_vars[i].name, ecFEF2575prctecFEV1_vars[i].name),
            (AR_vars[i].name, O2SatFFA_vars[i].name),
            (O2SatFFA_vars[i].name, UO2Sat_vars[i].name),
            (IA_vars[i].name, UO2Sat_vars[i].name),
            (UO2Sat_vars[i].name, O2Sat_vars[i].name),
        ]

    network += [(AR_vars[i - 1].name, AR_vars[i].name) for i in range(1, n)]

    model = BayesianNetwork(network)

    model.add_cpds(
        # Shared
        prior_hfev1,
        prior_ho2sat,
        # Days priors and CPTs
        *AR_priors,
        *uecFEV1_cpts,
        *ecFEV1_cpts,
        *ecFEF2575prctecFEV1_cpts,
        *O2SatFFA_cpts,
        *IA_priors,
        *UO2Sat_cpts,
        *O2Sat_cpts,
    )

    if check_model:
        model.check_model()
    return (
        model,
        HFEV1,
        HO2Sat,
        AR_vars,
        uecFEV1_vars,
        ecFEV1_vars,
        O2SatFFA_vars,
        IA_vars,
        UO2Sat_vars,
        O2Sat_vars,
        ecFEF2575prctecFEV1_vars,
    )


def fev1_o2sat_fef2575_noise_n_days_model_temporal_ar_with_ar_change_variable(
    n,
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
    S,
    check_model=True,
):
    """
    No direct link between AR and IA

    The AR change factor is fixed to the cpt[:, :, ar_change_cpt_state] for all days
    """

    def create_var_for_day(var, day):
        var_ = deepcopy(var)
        var_.name = f"{var.name} day {day}"
        return var_

    AR_vars = [create_var_for_day(AR, i) for i in range(1, n + 1)]
    uecFEV1_vars = [create_var_for_day(uecFEV1, i) for i in range(1, n + 1)]
    ecFEV1_vars = [create_var_for_day(ecFEV1, i) for i in range(1, n + 1)]
    ecFEF2575prctecFEV1_vars = [
        create_var_for_day(ecFEF2575prctecFEV1, i) for i in range(1, n + 1)
    ]
    O2SatFFA_vars = [create_var_for_day(O2SatFFA, i) for i in range(1, n + 1)]
    IA_vars = [create_var_for_day(IA, i) for i in range(1, n + 1)]
    UO2Sat_vars = [create_var_for_day(UO2Sat, i) for i in range(1, n + 1)]
    O2Sat_vars = [create_var_for_day(O2Sat, i) for i in range(1, n + 1)]

    # Priors and CPTs
    # AR first day prior shall be uniform
    AR_priors = []
    AR_priors.append(
        TabularCPD(
            variable=AR_vars[0].name,
            variable_card=AR_vars[0].card,
            values=AR_vars[0].first_day_prior.reshape(AR_vars[0].card, -1),
            evidence=[],
            evidence_card=[],
        )
    )
    # For other days the cpt depends the number of days elapsed
    # For testing, n days elapsed = 1
    for i in range(1, n):
        print(f"{AR_vars[i].name}, cpt: {AR_vars[i].change_cpt.shape}")
        AR_priors.append(
            TabularCPD(
                variable=AR_vars[i].name,
                variable_card=AR_vars[i].card,
                values=AR_vars[i].change_cpt.reshape(AR_vars[i].card, -1),
                evidence=[AR_vars[i - 1].name, S.name],
                evidence_card=[AR_vars[i - 1].card, S.card],
            )
        )

    uecFEV1_cpts = [
        build_pgmpy_ecfev1_cpt(uecFEV1_, HFEV1, AR_)
        for uecFEV1_, AR_ in zip(uecFEV1_vars, AR_vars)
    ]
    ecFEV1_cpts = [
        build_pgmpy_ecfev1_noise_cpt(ecFEV1_, uecFEV1_)
        for ecFEV1_, uecFEV1_ in zip(ecFEV1_vars, uecFEV1_vars)
    ]
    ecFEF2575prctecFEV1_cpts = [
        build_pgmpy_ecfef2575prctecfev1_cpt(
            ecFEF2575prctecFEV1_,
            AR_,
        )
        for ecFEF2575prctecFEV1_, AR_ in zip(ecFEF2575prctecFEV1_vars, AR_vars)
    ]
    O2SatFFA_cpts = [
        build_pgmpy_o2satffa_cpt(O2SatFFA_, HO2Sat, AR_)
        for O2SatFFA_, AR_ in zip(O2SatFFA_vars, AR_vars)
    ]
    IA_priors = [build_pgmpy_ia_prior(IA_) for IA_ in IA_vars]
    UO2Sat_cpts = [
        build_pgmpy_uo2sat_cpt(UO2Sat_, O2SatFFA_, IA_)
        for UO2Sat_, O2SatFFA_, IA_ in zip(UO2Sat_vars, O2SatFFA_vars, IA_vars)
    ]
    O2Sat_cpts = [
        build_pgmpy_o2sat_cpt(O2Sat_, UO2Sat_)
        for O2Sat_, UO2Sat_ in zip(O2Sat_vars, UO2Sat_vars)
    ]

    # Shared priors
    prior_hfev1 = build_pgmpy_hfev1_prior(HFEV1)
    prior_ho2sat = build_pgmpy_ho2sat_prior(HO2Sat)
    prior_S = build_pgmpy_ar_change_var_prior(S)

    # Make sure the days are at the same location in all variable lists
    for i in range(1, n + 1):
        assert AR_vars[i - 1].name == f"{AR.name} day {i}"
        assert uecFEV1_vars[i - 1].name == f"{uecFEV1.name} day {i}"
        assert ecFEV1_vars[i - 1].name == f"{ecFEV1.name} day {i}"
        assert (
            ecFEF2575prctecFEV1_vars[i - 1].name
            == f"{ecFEF2575prctecFEV1.name} day {i}"
        )
        assert O2SatFFA_vars[i - 1].name == f"{O2SatFFA.name} day {i}"
        assert IA_vars[i - 1].name == f"{IA.name} day {i}"
        assert UO2Sat_vars[i - 1].name == f"{UO2Sat.name} day {i}"
        assert O2Sat_vars[i - 1].name == f"{O2Sat.name} day {i}"

    network = []
    # HFEV1 and HO2Sat are shared between days
    for i in range(0, n):
        network += [
            (HFEV1.name, uecFEV1_vars[i].name),
            (HO2Sat.name, O2SatFFA_vars[i].name),
            (AR_vars[i].name, uecFEV1_vars[i].name),
            (uecFEV1_vars[i].name, ecFEV1_vars[i].name),
            (AR_vars[i].name, ecFEF2575prctecFEV1_vars[i].name),
            (AR_vars[i].name, O2SatFFA_vars[i].name),
            (O2SatFFA_vars[i].name, UO2Sat_vars[i].name),
            (IA_vars[i].name, UO2Sat_vars[i].name),
            (UO2Sat_vars[i].name, O2Sat_vars[i].name),
        ]

    # Deal with AR variables
    network += [(AR_vars[i - 1].name, AR_vars[i].name) for i in range(1, n)]
    network += [(S.name, AR_vars[i].name) for i in range(1, n)]

    print(network)

    model = BayesianNetwork(network)

    model.add_cpds(
        # Shared
        prior_hfev1,
        prior_ho2sat,
        prior_S,
        # Days priors and CPTs
        *AR_priors,
        *uecFEV1_cpts,
        *ecFEV1_cpts,
        *ecFEF2575prctecFEV1_cpts,
        *O2SatFFA_cpts,
        *IA_priors,
        *UO2Sat_cpts,
        *O2Sat_cpts,
    )

    # print('nodes', model.nodes())
    # print("cpds", model.get_cpds())

    if check_model:
        model.check_model()
    return (
        model,
        HFEV1,
        HO2Sat,
        AR_vars,
        uecFEV1_vars,
        ecFEV1_vars,
        O2SatFFA_vars,
        IA_vars,
        UO2Sat_vars,
        O2Sat_vars,
        ecFEF2575prctecFEV1_vars,
        S,
    )


def fev1_fef2575_o2sat_n_days_noise_factor_graph(
    n,
    HFEV1,
    uecFEV1,
    ecFEV1,
    AR,
    # HO2Sat,
    # O2SatFFA,
    # IA,
    # UO2Sat,
    # O2Sat,
    ecFEF2575prctecFEV1,
    ar_change_cpt_state=-1,
    check_model=True,
):
    """
    AR and IA have no direct link in this model
    """

    def create_var_for_day(var, day):
        var_ = deepcopy(var)
        var_.name = f"{var.name} day {day}"
        return var_

    AR_vars = [create_var_for_day(AR, i) for i in range(1, n + 1)]
    uecFEV1_vars = [create_var_for_day(uecFEV1, i) for i in range(1, n + 1)]
    ecFEV1_vars = [create_var_for_day(ecFEV1, i) for i in range(1, n + 1)]
    ecFEF2575prctecFEV1_vars = [
        create_var_for_day(ecFEF2575prctecFEV1, i) for i in range(1, n + 1)
    ]
    # O2SatFFA_vars = [create_var_for_day(O2SatFFA, i) for i in range(1, n + 1)]
    # IA_vars = [create_var_for_day(IA, i) for i in range(1, n + 1)]
    # UO2Sat_vars = [create_var_for_day(UO2Sat, i) for i in range(1, n + 1)]
    # O2Sat_vars = [create_var_for_day(O2Sat, i) for i in range(1, n + 1)]

    # Priors and CPTs
    # AR first day prior shall be uniform
    AR_priors = []
    AR_priors.append(
        DiscreteFactor(
            [AR_vars[0].name],
            [AR_vars[0].card],
            AR_vars[0].first_day_prior.reshape(AR_vars[0].card, -1),
        )
    )
    # For other days the cpt depends the number of days elapsed
    # For testing, n days elapsed = 1
    for i in range(1, n):
        AR_priors.append(
            DiscreteFactor(
                [AR_vars[i].name, AR_vars[i - 1].name],
                [AR_vars[i].card, AR_vars[i - 1].card],
                AR_vars[i].change_cpt[:, :, ar_change_cpt_state],
            )
        )

    uecFEV1_cpts = [
        build_pgmpy_ecfev1_factor_fn(uecFEV1_, HFEV1, AR_)
        for uecFEV1_, AR_ in zip(uecFEV1_vars, AR_vars)
    ]
    ecFEV1_cpts = [
        build_pgmpy_ecfev1_noise_factor_fn(ecFEV1_, uecFEV1_)
        for ecFEV1_, uecFEV1_ in zip(ecFEV1_vars, uecFEV1_vars)
    ]
    ecFEF2575prctecFEV1_cpts = [
        build_pgmpy_ecfef2575prctecfev1_factor_fn(
            ecFEF2575prctecFEV1_,
            AR_,
        )
        for ecFEF2575prctecFEV1_, AR_ in zip(ecFEF2575prctecFEV1_vars, AR_vars)
    ]
    # O2SatFFA_cpts = [
    #     build_pgmpy_o2satffa_factor_fn(O2SatFFA_, HO2Sat, AR_)
    #     for O2SatFFA_, AR_ in zip(O2SatFFA_vars, AR_vars)
    # ]
    # IA_priors = [build_pgmpy_ia_factor_fn(IA_) for IA_ in IA_vars]
    # UO2Sat_cpts = [
    #     build_pgmpy_uo2sat_factor_fn(UO2Sat_, O2SatFFA_, IA_)
    #     for UO2Sat_, O2SatFFA_, IA_ in zip(UO2Sat_vars, O2SatFFA_vars, IA_vars)
    # ]
    # O2Sat_cpts = [
    #     build_pgmpy_o2sat_factor_fn(O2Sat_, UO2Sat_)
    #     for O2Sat_, UO2Sat_ in zip(O2Sat_vars, UO2Sat_vars)
    # ]

    # Make sure the days are at the same location in all variable lists
    for i in range(1, n + 1):
        assert AR_vars[i - 1].name == f"{AR.name} day {i}"
        assert uecFEV1_vars[i - 1].name == f"{uecFEV1.name} day {i}"
        assert ecFEV1_vars[i - 1].name == f"{ecFEV1.name} day {i}"
        assert (
            ecFEF2575prctecFEV1_vars[i - 1].name
            == f"{ecFEF2575prctecFEV1.name} day {i}"
        )
        # assert O2SatFFA_vars[i - 1].name == f"{O2SatFFA.name} day {i}"
        # assert IA_vars[i - 1].name == f"{IA.name} day {i}"
        # assert UO2Sat_vars[i - 1].name == f"{UO2Sat.name} day {i}"
        # assert O2Sat_vars[i - 1].name == f"{O2Sat.name} day {i}"

    prior_hfev1 = build_pgmpy_hfev1_factor_fn(HFEV1)
    # prior_ho2sat = build_pgmpy_ho2sat_factor_fn(HO2Sat)

    G = FactorGraph()
    nodes = []
    for i in range(0, n):
        nodes += [
            uecFEV1_vars[i].name,
            ecFEV1_vars[i].name,
            AR_vars[i].name,
            # O2SatFFA_vars[i].name,
            # IA_vars[i].name,
            # UO2Sat_vars[i].name,
            # O2Sat_vars[i].name,
            ecFEF2575prctecFEV1_vars[i].name,
        ]
    G.add_nodes_from(
        [
            HFEV1.name,
            # HO2Sat.name,
        ]
        + nodes
    )
    G.add_factors(
        prior_hfev1,
        # prior_ho2sat,
        *uecFEV1_cpts,
        *ecFEV1_cpts,
        *ecFEF2575prctecFEV1_cpts,
        *AR_priors,
        # *O2SatFFA_cpts,
        # *IA_priors,
        # *UO2Sat_cpts,
        # *O2Sat_cpts,
    )
    inner_links = []
    for i in range(0, n):
        inner_links += [
            (HFEV1.name, uecFEV1_cpts[i]),
            (AR_priors[i], AR_vars[i].name),
            (AR_vars[i].name, uecFEV1_cpts[i]),
            (AR_vars[i].name, ecFEF2575prctecFEV1_cpts[i]),
            # (AR_vars[i].name, O2SatFFA_cpts[i]),
            # (HO2Sat.name, O2SatFFA_cpts[i]),
            # (O2SatFFA_cpts[i], O2SatFFA_vars[i].name),
            # (O2SatFFA_vars[i].name, UO2Sat_cpts[i]),
            # (IA_priors[i], IA_vars[i].name),
            # (IA_vars[i].name, UO2Sat_cpts[i]),
            # (UO2Sat_cpts[i], UO2Sat_vars[i].name),
            # (UO2Sat_vars[i].name, O2Sat_cpts[i]),
            # (O2Sat_cpts[i], O2Sat_vars[i].name),
            (uecFEV1_cpts[i], uecFEV1_vars[i].name),
            (uecFEV1_vars[i].name, ecFEV1_cpts[i]),
            (ecFEV1_cpts[i], ecFEV1_vars[i].name),
            (ecFEF2575prctecFEV1_cpts[i], ecFEF2575prctecFEV1_vars[i].name),
        ]
    between_days_links = []
    for i in range(1, n):
        between_days_links += [(AR_vars[i - 1].name, AR_priors[i])]

    G.add_edges_from(
        [
            (prior_hfev1, HFEV1.name),
            # (prior_ho2sat, HO2Sat.name),
        ]
        + inner_links
        + between_days_links
    )

    if check_model:
        assert G.check_model() == True
        print("Model is valid")

    return (
        G,
        HFEV1,
        # HO2Sat,
        AR_vars,
        uecFEV1_vars,
        ecFEV1_vars,
        # O2SatFFA_vars,
        # IA_vars,
        # UO2Sat_vars,
        # O2Sat_vars,
        ecFEF2575prctecFEV1_vars,
    )
