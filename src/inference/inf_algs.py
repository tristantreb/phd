from pgmpy.inference import BeliefPropagation, BeliefPropagationForFactorGraphs


def apply_pgmpy_bp(model):
    """
    Given a graphical model, returns the Belief Propagation class from pgmpy
    """
    return BeliefPropagation(model)


def apply_custom_bp(model):
    """
    Given a graphical model, returns the custom Belief Propagation class
    """
    return BeliefPropagationForFactorGraphs(model)
