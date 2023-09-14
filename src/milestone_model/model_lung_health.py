import sys

sys.path.append("../../")
sys.path.append("../data/")

import biology as bio
import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import BeliefPropagation
from pgmpy.models import BayesianNetwork

import model_helpers as mh


def set_fev1_prior(type, *args):
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


def build(healthy_FEV1_prior: object):
    U = mh.variableNode("Unblocked FEV1 (L)", 2, 6, 0.1, prior=healthy_FEV1_prior)
    C = mh.variableNode("Small airway availability (%)", 0.1, 1, 0.05)
    FEV1 = mh.variableNode("FEV1 (L)", 0.1, 6, 0.1)

    graph = BayesianNetwork([(U.name, FEV1.name), (C.name, FEV1.name)])

    cpt_fev1 = TabularCPD(
        variable=FEV1.name,
        variable_card=len(FEV1.bins) - 1,
        values=mh.calc_pgmpy_cpt(U, C, FEV1),
        evidence=[C.name, U.name],
        evidence_card=[len(C.bins) - 1, len(U.bins) - 1],
    )

    prior_c = TabularCPD(
        variable=C.name,
        variable_card=len(C.bins) - 1,
        values=C.prior,
        evidence=[],
        evidence_card=[],
    )

    prior_u = TabularCPD(
        variable=U.name,
        variable_card=len(U.bins) - 1,
        values=U.prior,
        evidence=[],
        evidence_card=[],
    )

    graph.add_cpds(cpt_fev1, prior_c, prior_u)

    graph.check_model()

    inference = BeliefPropagation(graph)
    return inference, FEV1, U, prior_u, C, prior_c


# In this model the small airway blockage and the lung damage are merge into one airway blockage variable
# This is done to simplify the model and to make it more intuitive
# In the future we will split this variables again (for the longidutinal model)
def build_healthy(healthy_FEV1_prior: object):
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
        values=mh.calc_pgmpy_cpt(HFEV1, AB, FEV1),
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


def infer(
    inference_model: BeliefPropagation,
    variables: tuple[mh.variableNode],
    evidences: tuple[tuple[mh.variableNode, float]],
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

    return inference_model.query(
        variables=var_names, evidence=evidences_binned, show_progress=False
    )
