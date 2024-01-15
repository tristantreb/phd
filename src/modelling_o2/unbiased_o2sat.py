import src.models.helpers as mh


def calc_cpt(UO2Sat: mh.variableNode, O2SatFFA: mh.variableNode, IA: mh.variableNode):
    """
    UO2Sat = O2SatFFA * (1-IA)
    Unbiased O2 saturation
    """
    return mh.calc_pgmpy_cpt_X_x_1_minus_Y(O2SatFFA, IA, UO2Sat)
