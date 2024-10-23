import src.models.helpers as mh
from pgmpy.factors.discrete import TabularCPD


def build_vevidence_cutset_conditioned_ar(
    AR: mh.VariableNode, state_n: int, curr_date, prev_date=None, next_date=None
):
    """
    Builds a virtual evidence for the airway resistance variable given the previous and next days

    Dates are datetime objects
    """
    prior = AR.get_virtual_message(state_n, curr_date, prev_date, next_date)
    return TabularCPD(AR.name, AR.card, prior.reshape(-1, 1))
