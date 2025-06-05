from pgmpy.inference import BeliefPropagation, BeliefPropagationWithMessagePassing


def apply_bayes_net_bp(model):
    """
    Given a graphical model, returns the Belief Propagation class from pgmpy
    """
    return BeliefPropagation(model)


def apply_factor_graph_bp(model, check_model=False):
    """
    Given a graphical model, returns the custom Belief Propagation class
    """
    return BeliefPropagationWithMessagePassing(model, check_model)
