"""
The functions in this file can be used to build the lung health models using the PGMPY library
"""

import time

import numpy as np

# PGMPY have been isolated in bayes_net_builders.py. This is tech debt.
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import BeliefPropagation
from pgmpy.models import BayesianNetwork

import src.models.bayes_net_builders as bayes_net_builders
import src.models.helpers as mh
import src.models.var_builders as var_builders


def set_LD_prior(fev1, pred_FEV1, pred_FEV1_std):
    """
    The high bound for lung damage is set to the 3rd max value of the FEV1
    """
    robust_max = np.sort(fev1)[-3]
    LD_high = robust_max / (pred_FEV1 - pred_FEV1_std)
    # Round to the nearest 0.05
    LD_high = round(LD_high * 20) / 20
    return {
        "type": "uniform + gaussian tail",
        "constant": LD_high,
        "sigma": pred_FEV1_std,
    }


def build_full_FEV1_side(
    HFEV1_prior,
    SAB_prior,
    LD_prior,
):
    """
    HFEV1
    LD
    UFEV1 = HFEV1 * (1 - LD)
    SAB
    FEV1 = UFEV1 * (1 - SAB)

    -> FEV1 = HFEV1 * (1 - LD) * (1 - SAB) -> can't solve because two unknowns
    """

    HFEV1_low = 1
    HFEV1_high = 6
    HFEV1 = mh.variableNode(
        "Healthy FEV1 (L)", HFEV1_low, HFEV1_high, 0.1, prior=HFEV1_prior
    )

    prior_HFEV1 = TabularCPD(
        variable=HFEV1.name,
        variable_card=len(HFEV1.bins),
        values=HFEV1.prior,
        evidence=[],
        evidence_card=[],
    )

    LD_low = 0
    LD_high = 0.8
    # It's not possible to live with >80% of global airway blockage (LD + SAB)
    LD = mh.variableNode("Lung Damage (%)", LD_low, LD_high, 0.05, prior=LD_prior)

    prior_LD = TabularCPD(
        variable=LD.name,
        variable_card=len(LD.bins),
        values=LD.prior,
        evidence=[],
        evidence_card=[],
    )

    UFEV1_low = HFEV1_low * (1 - LD_high)  # 0.2
    UFEV1_high = HFEV1_high * (1 - LD_low)  # 6
    UFEV1 = mh.variableNode("Unblocked FEV1 (L)", UFEV1_low, UFEV1_high, 0.1)

    cpt_UFEV1 = TabularCPD(
        variable=UFEV1.name,
        variable_card=len(UFEV1.bins),
        values=mh.calc_pgmpy_cpt_X_x_1_minus_Y(HFEV1, LD, UFEV1),
        evidence=[LD.name, HFEV1.name],
        evidence_card=[len(LD.bins), len(HFEV1.bins)],
    )

    # It's not possible to live with >80% of global airway blockage (LD + SAB)
    SAB_low = 0
    SAB_high = 0.8
    SAB = mh.variableNode(
        "Small Airway Blockage (%)",
        SAB_low,
        SAB_high,
        0.05,
        prior=SAB_prior,
    )

    prior_SAB = TabularCPD(
        variable=SAB.name,
        variable_card=len(SAB.bins),
        values=SAB.prior,
        evidence=[],
        evidence_card=[],
    )

    # Min observed to 0.4 in our data. Putting 0.1 for now
    FEV1_low = min(0.1, UFEV1_low * (1 - SAB_high))  # 0.04
    FEV1_high = UFEV1_high * (1 - SAB_low)  # 6
    FEV1 = mh.variableNode("FEV1 (L)", FEV1_low, FEV1_high, 0.1)

    cpt_FEV1 = TabularCPD(
        variable=FEV1.name,
        variable_card=len(FEV1.bins),
        values=mh.calc_pgmpy_cpt_X_x_1_minus_Y(UFEV1, SAB, FEV1),
        evidence=[SAB.name, UFEV1.name],
        evidence_card=[len(SAB.bins), len(UFEV1.bins)],
    )

    model = BayesianNetwork(
        [
            (HFEV1.name, UFEV1.name),
            (LD.name, UFEV1.name),
            (UFEV1.name, FEV1.name),
            (SAB.name, FEV1.name),
        ]
    )

    model.add_cpds(prior_HFEV1, prior_LD, cpt_UFEV1, prior_SAB, cpt_FEV1)

    model.check_model()

    inf_alg = BeliefPropagation(model)
    return model, inf_alg, HFEV1, LD, UFEV1, SAB, FEV1


def build_HFEV1_AB_FEV1(HFEV1_prior: object):
    """
    In this model the small airway blockage and the lung damage are merge into one airway blockage variable
    This is done to simplify the model and to make it more intuitive
    In the future we will split this variables again (for the longidutinal model)

    HFEV1
    AB
    FEV1 = HFEV1 * (1 - AB)
    """
    print("*** Building lung model with HFEV1 and AB ***")
    # The Heatlhy FEV1 takes the input prior distribution and truncates it in the interval [0.1,6)
    HFEV1 = mh.variableNode("Healthy FEV1 (L)", 1, 6, 0.1, HFEV1_prior)
    # It's not possible to live with >80% of airway blockage
    AB = mh.variableNode("Airway Degradation", 0, 0.8, 0.05, prior={"type": "uniform"})
    FEV1 = mh.variableNode("FEV1 (L)", 0.1, 6, 0.1, None)

    model = BayesianNetwork([(HFEV1.name, FEV1.name), (AB.name, FEV1.name)])

    cpt_fev1 = TabularCPD(
        variable=FEV1.name,
        variable_card=len(FEV1.bins),
        values=mh.calc_pgmpy_cpt_X_x_1_minus_Y(HFEV1, AB, FEV1),
        evidence=[HFEV1.name, AB.name],
        evidence_card=[len(HFEV1.bins), len(AB.bins)],
    )

    prior_ab = TabularCPD(
        variable=AB.name,
        variable_card=len(AB.bins),
        values=AB.prior,
        evidence=[],
        evidence_card=[],
    )

    prior_u = TabularCPD(
        variable=HFEV1.name,
        variable_card=len(HFEV1.bins),
        values=HFEV1.prior,
        evidence=[],
        evidence_card=[],
    )

    model.add_cpds(cpt_fev1, prior_ab, prior_u)

    model.check_model()

    inf_alg = BeliefPropagation(model)

    return model, inf_alg, FEV1, HFEV1, AB


def build_FEV1_O2_point_in_time_model(hfev1_prior, ho2sat_prior):
    """
    This is a point in time model with
    FEV1 = HFEV1 * (1-AR)
    O2SatFFA = HO2Sat * drop_func(AR)

    The model is the same as build_HFEV1_AB_FEV1(), with Airway Blockage renamed to Airway Resistance.
    """
    print("*** Building FEV1 and O2 point in time model ***")

    # The Heatlhy FEV1 takes the input prior distribution and truncates it in the interval [0.1,6)
    HFEV1 = mh.variableNode("Healthy FEV1 (L)", 1, 6, 0.05, prior=hfev1_prior)
    AR = mh.variableNode("Airway Resistance (%)", 0, 90, 1, prior={"type": "uniform"})
    ecFEV1 = mh.variableNode("FEV1 (L)", 0, 6, 0.05, prior=None)
    # HO2Sat = mh.variableNode(
    #     "Healthy O2 Saturation (%)", 90, 100, 1, prior=ho2sat_prior
    # )
    # O2SatFFA = mh.variableNode(
    #     "O2 Sat if fully functional alveoli (%)", 70, 100, 1, prior=None
    # )

    prior_hfev1 = TabularCPD(
        variable=HFEV1.name,
        variable_card=len(HFEV1.bins),
        values=HFEV1.prior,
        evidence=[],
        evidence_card=[],
    )
    # prior_ho2sat = TabularCPD(
    #     variable=HO2Sat.name,
    #     variable_card=len(HO2Sat.bins),
    #     values=HO2Sat.prior,
    #     evidence=[],
    #     evidence_card=[],
    # )
    prior_ar = TabularCPD(
        variable=AR.name,
        variable_card=len(AR.bins),
        values=AR.prior,
        evidence=[],
        evidence_card=[],
    )
    print(prior_ar)
    cpt_fev1 = TabularCPD(
        variable=ecFEV1.name,
        variable_card=len(ecFEV1.bins),
        values=mh.calc_pgmpy_cpt_X_x_1_minus_Y(HFEV1, AR, ecFEV1),
        evidence=[HFEV1.name, AR.name],
        evidence_card=[len(HFEV1.bins), len(AR.bins)],
    )
    # cpt_o2_sat_ffa = TabularCPD(
    #     variable=O2SatFFA.name,
    #     variable_card=len(O2SatFFA.bins),
    #     values=o2satffa_factor.calc_cpt_O2SatFFA_HO2Sat_AR(
    #         O2SatFFA, HO2Sat, AR, debug=False
    #     ),
    #     evidence=[HO2Sat.name, AR.name],
    #     evidence_card=[len(HO2Sat.bins), len(AR.bins)],
    # )
    print(f"Time to build variables: {time.time() - tic}")
    tic = time.time()

    model = BayesianNetwork(
        [
            (HFEV1.name, ecFEV1.name),
            (AR.name, ecFEV1.name),
            # (HO2Sat.name, O2SatFFA.name),
            # (AR.name, O2SatFFA.name),
        ]
    )

    model.add_cpds(cpt_fev1, AR.prior, prior_hfev1)
    # model.add_cpds(cpt_fev1, prior_ar, prior_hfev1, prior_ho2sat, cpt_o2_sat_ffa)

    model.check_model()
    inf_alg = BeliefPropagation(model)
    print(f"Time to build model: {time.time() - tic}")
    return (model, inf_alg, HFEV1, ecFEV1, AR)
    # return (model, inf_alg, HFEV1, ecFEV1, HO2Sat, O2SatFFA, AR)


def build_longitudinal_FEV1_side(
    n,
    HFEV1_prior={"type": "uniform"},
    SAB_prior={"type": "uniform"},
    LD_prior={"type": "uniform"},
):
    """
    n: number of time points

    HFEV1: shared
    LD: shared across all time points, to model that it's constant at the time scale we're considering
    UFEV1 = HFEV1 * (1 - LD): shared
    SAB: one variable per time point
    FEV1 = UFEV1 * (1 - SAB): one variable per time point
    """
    print(
        "*** Building the longitudinal model with LD as shared variable across time ***"
    )

    # Variables shared across tmie
    HFEV1_low = 1
    HFEV1_high = 6
    HFEV1 = mh.variableNode(
        "Healthy FEV1 (L)", HFEV1_low, HFEV1_high, 0.1, prior=HFEV1_prior
    )

    prior_HFEV1 = TabularCPD(
        variable=HFEV1.name,
        variable_card=len(HFEV1.bins),
        values=HFEV1.prior,
        evidence=[],
        evidence_card=[],
    )

    LD_low = 0
    LD_high = 0.8
    # It's not possible to live with >80% of global airway blockage (LD + SAB)
    LD = mh.variableNode("Lung Damage (%)", LD_low, LD_high, 0.05, prior=LD_prior)

    prior_LD = TabularCPD(
        variable=LD.name,
        variable_card=len(LD.bins),
        values=LD.prior,
        evidence=[],
        evidence_card=[],
    )

    UFEV1_low = HFEV1_low * (1 - LD_high)  # 0.2
    UFEV1_high = HFEV1_high * (1 - LD_low)  # 6
    UFEV1 = mh.variableNode("Unblocked FEV1 (L)", UFEV1_low, UFEV1_high, 0.1)

    cpt_UFEV1 = TabularCPD(
        variable=UFEV1.name,
        variable_card=len(UFEV1.bins),
        values=mh.calc_pgmpy_cpt_X_x_1_minus_Y(HFEV1, LD, UFEV1),
        evidence=[LD.name, HFEV1.name],
        evidence_card=[len(LD.bins), len(HFEV1.bins)],
    )

    # One variable per time point.
    SAB_list = []
    FEV1_list = []
    SAB_FEV1_pairs_list = []
    UFEV1_FEV1_pairs_list = []
    SAB_priors = []
    FEV1_cpts = []

    # TODO: improve efficiency by creating sharing one variable node and simply changing the name
    for i in range(n):
        # It's not possible to live with >80% of global airway blockage (LD + SAB)
        SAB_low = 0
        SAB_high = 0.8
        SAB_i = mh.variableNode(
            f"Small Airway Blockage {i} (%)",
            SAB_low,
            SAB_high,
            0.05,
            prior=SAB_prior,
        )

        prior_SAB_i = TabularCPD(
            variable=SAB_i.name,
            variable_card=len(SAB_i.bins),
            values=SAB_i.prior,
            evidence=[],
            evidence_card=[],
        )

        # Min observed to 0.4 in our data. Putting 0.1 for now
        FEV1_low = min(0.1, UFEV1_low * (1 - SAB_high))  # 0.04
        FEV1_high = UFEV1_high * (1 - SAB_low)  # 6
        FEV1_i = mh.variableNode(f"FEV1 {i} (L)", FEV1_low, FEV1_high, 0.1)

        cpt_FEV1 = TabularCPD(
            variable=FEV1_i.name,
            variable_card=len(FEV1_i.bins),
            values=mh.calc_pgmpy_cpt_X_x_1_minus_Y(UFEV1, SAB_i, FEV1_i),
            evidence=[SAB_i.name, UFEV1.name],
            evidence_card=[len(SAB_i.bins), len(UFEV1.bins)],
        )

        SAB_list.append(SAB_i)
        FEV1_list.append(FEV1_i)
        SAB_FEV1_pairs_list.append((SAB_i.name, FEV1_i.name))
        UFEV1_FEV1_pairs_list.append((UFEV1.name, FEV1_i.name))
        SAB_priors.append(prior_SAB_i)
        FEV1_cpts.append(cpt_FEV1)

    model = BayesianNetwork(
        [
            (HFEV1.name, UFEV1.name),
            (LD.name, UFEV1.name),
        ]
        + SAB_FEV1_pairs_list
        + UFEV1_FEV1_pairs_list
    )

    model.add_cpds(prior_HFEV1, prior_LD, cpt_UFEV1, *SAB_priors, *FEV1_cpts)

    model.check_model()

    inf_alg = BeliefPropagation(model)
    return (
        model,
        inf_alg,
        HFEV1,
        LD,
        UFEV1,
        SAB_list,
        FEV1_list,
    )


def o2sat_fev1_point_in_time_model(height, age, sex):
    """
    Point in time model with full FEV1 and O2Sat sides

    There is no factor linking AR and IA in this model
    IA and AR's priors are uniform
    """

    (
        HFEV1,
        ecFEV1,
        AR,
        HO2Sat,
        O2SatFFA,
        IA,
        UO2Sat,
        O2Sat,
    ) = var_builders.o2sat_fev1_point_in_time(height, age, sex)

    (
        model,
        inf_alg,
    ) = bayes_net_builders.fev1_o2sat_point_in_time_model(
        HFEV1, ecFEV1, AR, HO2Sat, O2SatFFA, IA, UO2Sat, O2Sat
    )
    return model, inf_alg, HFEV1, ecFEV1, AR, HO2Sat, O2SatFFA, IA, UO2Sat, O2Sat


def o2sat_fev1_point_in_time_model_cf_priors(height, age, sex, ar_prior, ia_prior):
    """
    Point in time model with full FEV1 and O2Sat sides

    There is no factor linking AR and IA in this model
    The priors for AR, IA are learnt from the Breathe data
    """

    (
        HFEV1,
        ecFEV1,
        AR,
        HO2Sat,
        O2SatFFA,
        IA,
        UO2Sat,
        O2Sat,
    ) = var_builders.o2sat_fev1_point_in_time_cf_priors(
        height, age, sex, ar_prior, ia_prior
    )

    (
        model,
        inf_alg,
    ) = bayes_net_builders.fev1_o2sat_point_in_time_model(
        HFEV1, ecFEV1, AR, HO2Sat, O2SatFFA, IA, UO2Sat, O2Sat
    )
    return model, inf_alg, HFEV1, ecFEV1, AR, HO2Sat, O2SatFFA, IA, UO2Sat, O2Sat


def o2sat_fev1_point_in_time_model_cf_ia_prior(height, age, sex):
    """
    Point in time model with full FEV1 and O2Sat sides

    There is no factor linking AR and IA in this model
    IA's prior is learnt from the Breathe data (heavy tailed)
    AR's prior is uniform
    """

    (
        HFEV1,
        ecFEV1,
        AR,
        HO2Sat,
        O2SatFFA,
        IA,
        UO2Sat,
        O2Sat,
    ) = var_builders.o2sat_fev1_point_in_time_cf_ia_prior(height, age, sex)

    (
        model,
        inf_alg,
    ) = bayes_net_builders.fev1_o2sat_point_in_time_model(
        HFEV1, ecFEV1, AR, HO2Sat, O2SatFFA, IA, UO2Sat, O2Sat
    )
    return model, inf_alg, HFEV1, ecFEV1, AR, HO2Sat, O2SatFFA, IA, UO2Sat, O2Sat


def o2sat_fev1_point_in_time_model_cf_priors_2(height, age, sex, ar_prior, cpd_ar_ia):
    """
    Point in time model with full FEV1 and O2Sat sides

    AR prior is given
    IA has no prior as it's caused by AR according to cpd_ar_ia
    """

    (
        HFEV1,
        ecFEV1,
        AR,
        HO2Sat,
        O2SatFFA,
        IA,
        UO2Sat,
        O2Sat,
    ) = var_builders.o2sat_fev1_point_in_time_model_ar_ia_factor_test(
        height, age, sex, ar_prior, cpd_ar_ia
    )

    (
        model,
        inf_alg,
    ) = bayes_net_builders.fev1_o2sat_point_in_time_model_2(
        HFEV1, ecFEV1, AR, HO2Sat, O2SatFFA, IA, UO2Sat, O2Sat
    )
    return model, inf_alg, HFEV1, ecFEV1, AR, HO2Sat, O2SatFFA, IA, UO2Sat, O2Sat
