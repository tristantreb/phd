import models.helpers as mh
from pgmpy.factors.discrete import TabularCPD


def build_vevidence_cutset_conditioned_ar(
    AR: mh.VariableNode,
    state_n: int,
    curr_date,
    prev_date=None,
    next_date=None,
    debug=False,
):
    """
    Builds a virtual evidence for the airway resistance variable given the previous and next days

    Dates are datetime objects
    """
    prior = AR.get_virtual_message(state_n, curr_date, prev_date, next_date, debug)
    return TabularCPD(AR.name, AR.card, prior.reshape(-1, 1))


def build_vevidence_cutset_conditioned_ar_with_shape_factor(
    AR: mh.VariableNode,
    state_n: int,
    curr_date,
    shape_factor,
    prev_date=None,
    next_date=None,
    n_days_consec=1,
    debug=False,
):
    """
    Builds a virtual evidence for the airway resistance variable given the previous and next days

    Dates are datetime objects
    """
    prior = AR.get_virtual_message_with_shape_factor(
        state_n, curr_date, shape_factor, prev_date, next_date, n_days_consec, debug
    )
    return TabularCPD(AR.name, AR.card, prior.reshape(-1, 1))
