from pgmpy.inference import BeliefPropagation


def apply_pgmpy_bp(model):
    """
    Given a graphical model, returns the Belief Propagation class from pgmpy
    """
    return BeliefPropagation(model)
