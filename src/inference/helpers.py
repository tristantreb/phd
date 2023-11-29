import numpy as np
import plotly.graph_objects as go
from pgmpy.inference import BeliefPropagation

import src.models.helpers as mh

# Set global value for tolerance.
# This to account for the rounding error: https://www.cs.drexel.edu/~jpopyack/Courses/CSP/Fa17/extras/Rounding/index.html#:~:text=Rounding%20(roundoff)%20error%20is%20a,word%20size%20used%20for%20integers.
TOL_GLOBAL = 1e-6
# Switch from 1e-8 to 1e-6 to because got 0.9999999885510139 sum of probabilities for a model with AW


def infer(
    inference_alg: BeliefPropagation,
    variables: tuple[mh.variableNode],
    evidences: tuple[tuple[mh.variableNode, float]],
    show_progress=False,
    joint=True,
):
    """
    Runs an inference query against a given PGMPY inference model, variables, evidences
    :param inference_alg: The inference algorithm to use
    :param variables: The variables to query
    :param evidences: The evidences to use

    :return: The result of the inference
    """
    var_names = [var.name for var in variables]

    evidences_binned = dict()
    for [evidence_var, value] in evidences:
        [_bin, bin_idx] = get_bin_for_value(value, evidence_var)
        evidences_binned.update({evidence_var.name: bin_idx})

    # tic = time.time()
    query = inference_alg.query(
        variables=var_names,
        evidence=evidences_binned,
        show_progress=show_progress,
        joint=joint,
    )
    # print(f"Query took {time.time() - tic}s")

    return query


# Given an observation and an array of bins, this returns the bin that the value falls into
def get_bin_for_value(obs: float, var: mh.variableNode, tol=TOL_GLOBAL):
    # Center bins around value observed
    relative_bins = var.bins - obs - tol

    # Find the highest negative value of the bins relative to centered bins
    idx = np.where(relative_bins <= 0, relative_bins, -np.inf).argmax()

    (lower_idx, upper_idx) = var.bins_arr[idx]
    return ["[{}; {})".format(lower_idx, upper_idx), idx]


def plot_histogram(fig, Var: mh.variableNode, p, min, max, row, col, title=True):
    fig.add_trace(
        go.Histogram(
            x=Var.bins,
            y=p,
            histfunc="sum",  # Use 'sum' to represent pre-counted data
            xbins=dict(start=min, end=max, size=Var.bin_width),
        ),
        row=row,
        col=col,
    )
    fig.update_xaxes(title=Var.name if title else None, row=row, col=col)
    return -1
