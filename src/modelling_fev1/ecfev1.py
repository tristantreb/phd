import models.helpers as mh


def calc_cpt(
    ecFEV1: mh.VariableNode, HFEV1: mh.VariableNode, AR: mh.VariableNode, debug=False
):
    """
    Computes the CPT for P(ecFEV1 | HFEV1, AR)
    """

    cpt = mh.calc_pgmpy_cpt_X_x_1_minus_Y(HFEV1, AR, ecFEV1, debug=debug)

    return cpt.reshape(ecFEV1.card, HFEV1.card, AR.card)
