import bp
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import BeliefPropagation
from pgmpy.models import BayesianNetwork

U = bp.variableNode("Unblocked FEV1 (L)", 2, 6, 0.1)
C = bp.variableNode("1-Small airway blockage (%)", 0.1, 1, 0.05)
FEV1 = bp.variableNode("FEV1 (L)", 0.2, 6, 0.1)

graph = BayesianNetwork([(U.name, FEV1.name), (C.name, FEV1.name)])

cpt_fev1 = TabularCPD(
    variable=FEV1.name,
    variable_card=len(FEV1.bins) - 1,
    values=bp.calc_pgmpy_cpt(U, C, FEV1),
    evidence=[C.name, U.name],
    evidence_card=[len(C.bins) - 1, len(U.bins) - 1],
)

prior_b = TabularCPD(
    variable=C.name,
    variable_card=len(C.bins) - 1,
    values=C.marginal(C),
    evidence=[],
    evidence_card=[],
)

prior_u = TabularCPD(
    variable=U.name,
    variable_card=len(U.bins) - 1,
    values=U.marginal(U),
    evidence=[],
    evidence_card=[],
)

graph.add_cpds(cpt_fev1, prior_b, prior_u)

graph.check_model()

inference = BeliefPropagation(graph)
