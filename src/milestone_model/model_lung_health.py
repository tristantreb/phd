from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import BeliefPropagation
from pgmpy.models import BayesianNetwork

import model_helpers as mh


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
    # The Heatlhy FEV1 takes the input prior distribution and truncates it in the interval [2,6)
    HFEV1 = mh.variableNode("Healthy FEV1 (L)", 1, 6, 0.1, prior=healthy_FEV1_prior)
    # It's not possible to live with 0-20% of airway availability
    Av = mh.variableNode("Airway availability", 0.2, 1, 0.05)
    FEV1 = mh.variableNode("FEV1 (L)", 0.1, 6, 0.1)

    graph = BayesianNetwork([(HFEV1.name, FEV1.name), (Av.name, FEV1.name)])

    cpt_fev1 = TabularCPD(
        variable=FEV1.name,
        variable_card=len(FEV1.bins) - 1,
        values=mh.calc_pgmpy_cpt(HFEV1, Av, FEV1),
        evidence=[Av.name, HFEV1.name],
        evidence_card=[len(Av.bins) - 1, len(HFEV1.bins) - 1],
    )

    prior_av = TabularCPD(
        variable=Av.name,
        variable_card=len(Av.bins) - 1,
        values=Av.prior,
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

    graph.add_cpds(cpt_fev1, prior_av, prior_u)

    graph.check_model()

    inference = BeliefPropagation(graph)
    return inference, FEV1, HFEV1, prior_u, Av, prior_av
