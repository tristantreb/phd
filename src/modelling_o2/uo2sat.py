import src.models.helpers as mh


def calc_cpt(UO2Sat: mh.VariableNode, O2SatFFA: mh.VariableNode, IA: mh.VariableNode):
    """
    UO2Sat = O2SatFFA * (1-IA)
    UO2Sat: Underling O2 saturation
    O2SatFFA: O2 saturation if fully functional alveoli
    IA: Inactive alveoli
    """
    return mh.calc_pgmpy_cpt_X_x_1_minus_Y(O2SatFFA, IA, UO2Sat)
