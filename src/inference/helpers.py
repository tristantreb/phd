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
    joint=False,
):
    """
    Runs an inference query against a given PGMPY inference model, variables, evidences
    :param inference_alg: The inference algorithm to use
    :param variables: The variables to query
    :param evidences: The evidences to use

    :return: The result of the inference
    """

    evidences_binned = dict()
    for [evidence_var, value] in evidences:
        [_bin, bin_idx] = get_bin_for_value(value, evidence_var)
        evidences_binned.update({evidence_var.name: bin_idx})

    query = inference_alg.query(
        variables=list(map(lambda v: v.name, variables)),
        evidence=evidences_binned,
        show_progress=show_progress,
        joint=joint,
    )

    return query


# Given an observation and an array of bins, this returns the bin that the value falls into
def get_bin_for_value(obs: float, var: mh.variableNode, tol=TOL_GLOBAL):
    # Center bins around value observed
    relative_bins = var.bins - obs - tol

    # Find the highest negative value of the bins relative to centered bins
    idx = np.where(relative_bins <= 0, relative_bins, -np.inf).argmax()

    (lower_idx, upper_idx) = var.bins_arr[idx]
    return "[{}; {})".format(lower_idx, upper_idx), idx


def plot_histogram(fig, Var: mh.variableNode, p, xmin, xmax, row, col, title=True):
    fig.add_trace(
        go.Histogram(
            x=Var.bins,
            y=p,
            histfunc="sum",  # Use 'sum' to represent pre-counted data
            xbins=dict(start=xmin, end=xmax, size=Var.bin_width),
        ),
        row=row,
        col=col,
    )
    fig.update_xaxes(
        range=[xmin, xmax],
        nticks=20,
        title=Var.name if title else None,
        row=row,
        col=col,
    )
    # Add one specific tick label on the x axis
    # fig.update_xaxes(tickvals=[Var.get_mean(p)], row=row, col=col)
    # Add distribution's mean as an annotation with small font size
    fig.add_annotation(
        x=Var.get_mean(p),
        y=max(p) * 1.1,
        text=f"{Var.get_mean(p):.2f}",
        showarrow=False,
        font=dict(size=6),
        row=row,
        col=col,
    )
    return -1
