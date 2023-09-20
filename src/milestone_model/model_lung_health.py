import sys

sys.path.append("../../")
sys.path.append("../data/")

import biology as bio
import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import BeliefPropagation
from pgmpy.models import BayesianNetwork

import model_helpers as mh
import time


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


def set_HFEV1_prior(type, *args):
    """
    type: "uniform" or "gaussian"
    args: if type == "uniform", no args
    args: if type == "gaussian", height (cm), age (yr), "Male" or "Female"
    """
    if type == "uniform":
        return {"type": "uniform"}
    elif type == "gaussian":
        set_height = args[0]
        set_age = args[1]
        set_sex = args[2]
        FEV1 = bio.calc_predicted_fev1(set_height, set_age, set_sex)
        pred_FEV1 = FEV1["Predicted FEV1"]
        pred_FEV1_std = FEV1["std"]
        return {"type": "gaussian", "mu": pred_FEV1, "sigma": pred_FEV1_std}
    else:
        raise ValueError("Invalid type (should be uniform or gaussian)")


def build_full_FEV1_side(
    HFEV1_prior={"type": "uniform"},
    SAB_prior={"type": "uniform"},
    LD_prior={"type": "uniform"},
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
        variable_card=len(HFEV1.bins) - 1,
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
        variable_card=len(LD.bins) - 1,
        values=LD.prior,
        evidence=[],
        evidence_card=[],
    )

    UFEV1_low = HFEV1_low * (1 - LD_high)  # 0.2
    UFEV1_high = HFEV1_high * (1 - LD_low)  # 6
    UFEV1 = mh.variableNode("Unblocked FEV1 (L)", UFEV1_low, UFEV1_high, 0.1)

    cpt_UFEV1 = TabularCPD(
        variable=UFEV1.name,
        variable_card=len(UFEV1.bins) - 1,
        values=mh.calc_pgmpy_cpt_X_x_1_minus_Y(HFEV1, LD, UFEV1),
        evidence=[LD.name, HFEV1.name],
        evidence_card=[len(LD.bins) - 1, len(HFEV1.bins) - 1],
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
        variable_card=len(SAB.bins) - 1,
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
        variable_card=len(FEV1.bins) - 1,
        values=mh.calc_pgmpy_cpt_X_x_1_minus_Y(UFEV1, SAB, FEV1),
        evidence=[SAB.name, UFEV1.name],
        evidence_card=[len(SAB.bins) - 1, len(UFEV1.bins) - 1],
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

    inference = BeliefPropagation(model)
    return inference, HFEV1, prior_HFEV1, LD, prior_LD, UFEV1, SAB, prior_SAB, FEV1


# In this model the small airway blockage and the lung damage are merge into one airway blockage variable
# This is done to simplify the model and to make it more intuitive
# In the future we will split this variables again (for the longidutinal model)
def build_HFEV1_AB_FEV1(healthy_FEV1_prior: object):
    """
    HFEV1
    AB
    FEV1 = HFEV1 * (1 - AB)
    """
    print("*** Building lung model with HFEV1 and AB ***")
    # The Heatlhy FEV1 takes the input prior distribution and truncates it in the interval [2,6)
    HFEV1 = mh.variableNode("Healthy FEV1 (L)", 1, 6, 0.1, prior=healthy_FEV1_prior)
    # It's not possible to live with >80% of airway blockage
    AB = mh.variableNode("Airway Blockage", 0, 0.8, 0.05)
    FEV1 = mh.variableNode("FEV1 (L)", 0.1, 6, 0.1)

    graph = BayesianNetwork([(HFEV1.name, FEV1.name), (AB.name, FEV1.name)])

    cpt_fev1 = TabularCPD(
        variable=FEV1.name,
        variable_card=len(FEV1.bins) - 1,
        values=mh.calc_pgmpy_cpt_X_x_1_minus_Y(HFEV1, AB, FEV1),
        evidence=[AB.name, HFEV1.name],
        evidence_card=[len(AB.bins) - 1, len(HFEV1.bins) - 1],
    )

    prior_ab = TabularCPD(
        variable=AB.name,
        variable_card=len(AB.bins) - 1,
        values=AB.prior,
        evidence=[],
        evidence_card=[],
    )

    prior_u = TabularCPD(
        variable=HFEV1.name,
        variable_card=len(HFEV1.bins) - 1,
        values=HFEV1.prior,
        evidence=[],
        evidence_card=[],
    )

    graph.add_cpds(cpt_fev1, prior_ab, prior_u)

    graph.check_model()

    inference = BeliefPropagation(graph)
    return inference, FEV1, HFEV1, prior_u, AB, prior_ab


def build_o2_sat():
    UO2Sat = mh.variableNode("Unblocked O2 Sat (%)", 0.7, 1, 0.1)
    LD = mh.variableNode("Lung damage (%)", 0.2, 1, 0.1)

    graph = BayesianNetwork([(UO2Sat.name, LD.name)])

    cpt_u_o2 = TabularCPD(
        variable=UO2Sat.name,
        variable_card=len(UO2Sat.bins) - 1,
        values=[1, 1],
        evidence=[LD.name],
        evidence_card=[len(LD.bins) - 1],
    )

    prior_ld = TabularCPD(
        variable=LD.name,
        variable_card=len(LD.bins) - 1,
        values=LD.prior,
        evidence=[],
        evidence_card=[],
    )

    graph.add_cpds(cpt_u_o2, prior_ld)

    graph.check_model()

    inference = BeliefPropagation(graph)
    return inference, UO2Sat, LD, prior_ld


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
        variable_card=len(HFEV1.bins) - 1,
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
        variable_card=len(LD.bins) - 1,
        values=LD.prior,
        evidence=[],
        evidence_card=[],
    )

    UFEV1_low = HFEV1_low * (1 - LD_high)  # 0.2
    UFEV1_high = HFEV1_high * (1 - LD_low)  # 6
    UFEV1 = mh.variableNode("Unblocked FEV1 (L)", UFEV1_low, UFEV1_high, 0.1)

    cpt_UFEV1 = TabularCPD(
        variable=UFEV1.name,
        variable_card=len(UFEV1.bins) - 1,
        values=mh.calc_pgmpy_cpt_X_x_1_minus_Y(HFEV1, LD, UFEV1),
        evidence=[LD.name, HFEV1.name],
        evidence_card=[len(LD.bins) - 1, len(HFEV1.bins) - 1],
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
            variable_card=len(SAB_i.bins) - 1,
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
            variable_card=len(FEV1_i.bins) - 1,
            values=mh.calc_pgmpy_cpt_X_x_1_minus_Y(UFEV1, SAB_i, FEV1_i),
            evidence=[SAB_i.name, UFEV1.name],
            evidence_card=[len(SAB_i.bins) - 1, len(UFEV1.bins) - 1],
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

    inference = BeliefPropagation(model)
    return (
        model,
        inference,
        HFEV1,
        prior_HFEV1,
        LD,
        prior_LD,
        UFEV1,
        SAB_list,
        prior_SAB_i,
        FEV1_list,
    )


def infer(
    inference_model: BeliefPropagation,
    variables: tuple[mh.variableNode],
    evidences: tuple[tuple[mh.variableNode, float]],
    joint=True,
):
    """
    Runs an inference query against a given PGMPY inference model, variables, evidences
    :param inference_model: The inference model to use
    :param variables: The variables to query
    :param evidences: The evidences to use

    :return: The result of the inference
    """
    var_names = [var.name for var in variables]

    evidences_binned = dict()
    for [evidence_var, value] in evidences:
        [_bin, bin_idx] = mh.get_bin_for_value(value, evidence_var.bins)
        evidences_binned.update({evidence_var.name: bin_idx})

    tic = time.time()
    query = inference_model.query(
        variables=var_names, evidence=evidences_binned, show_progress=True, joint=joint
    )
    print(f"Query took {time.time() - tic}s")

    return query
