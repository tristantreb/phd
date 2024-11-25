import numpy as np
import plotly.graph_objects as go

import src.models.helpers as mh
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import BeliefPropagation, BeliefPropagationWithMessageParsing

# Set global value for tolerance.
# This to account for the rounding error: https://www.cs.drexel.edu/~jpopyack/Courses/CSP/Fa17/extras/Rounding/index.html#:~:text=Rounding%20(roundoff)%20error%20is%20a,word%20size%20used%20for%20integers.
TOL_GLOBAL = 1e-6
# Switch from 1e-8 to 1e-6 to because got 0.9999999885510139 sum of probabilities for a model with AW


def infer_on_factor_graph(
    inference_alg: BeliefPropagationWithMessageParsing,
    variables: tuple[mh.VariableNode],
    evidence: tuple[tuple[mh.VariableNode, float]],
    virtual_evidence=None,
    get_messages=False,
):
    """
    Runs an inference query against a given PGMPY inference model, variables, evidence
    :param inference_alg: The inference algorithm to use
    :param variables: The variables to query
    :param evidence: The evidence to use
    :param virtual_evidence: The virtual evidence to use, if exists it's a tuple[TabularCPD]

    :return: The result of the inference
    """
    evidence_binned = {
        evidence_var.name: evidence_var.get_bin_for_value(value, evidence_var)
        for evidence_var, value in evidence
    }

    res = inference_alg.query(
        variables=list(map(lambda v: v.name, variables)),
        evidence=evidence_binned,
        virtual_evidence=virtual_evidence,
        get_messages=get_messages,
    )
    return res


def infer(
    inference_alg: BeliefPropagation,
    variables: tuple[mh.VariableNode],
    evidence: tuple[tuple[mh.VariableNode, float]],
    show_progress=False,
    joint=False,
    debug=False,
):
    """
    Runs an inference query against a given PGMPY inference model, variables, evidence
    :param inference_alg: The inference algorithm to use
    :param variables: The variables to query
    :param evidence: The evidence to use

    :return: The result of the inference
    """
    variables = list(map(lambda v: v.name, variables))
    if debug:
        print(f"Variables to infer: {variables}")

    evidence_binned = {
        evidence_var.name: evidence_var.get_bin_for_value(value, evidence_var)
        for evidence_var, value in evidence
    }

    res = inference_alg.query(
        variables=variables,
        evidence=evidence_binned,
        show_progress=show_progress,
        joint=joint,
    )
    return res


def plot_histogram(
    fig,
    Var: mh.VariableNode,
    p,
    xmin,
    xmax,
    row,
    col,
    title=None,
    colour=None,
    annot=True,
    name=None,
    xlabels=True,
    clean_ticks=False,
):
    fig.add_trace(
        go.Histogram(
            x=Var.bins,
            y=p,
            histfunc="sum",  # Use 'sum' to represent pre-counted data
            xbins=dict(start=xmin, end=xmax, size=Var.bin_width),
            name=name,
        ),
        row=row,
        col=col,
    )
    if colour:
        fig.update_traces(marker_color=colour, row=row, col=col)

    # Hide x axis labels
    if not xlabels:
        fig.update_xaxes(showticklabels=False, row=row, col=col)
    fig.update_xaxes(
        range=[xmin, xmax],
        nticks=20,
        title=title,
        row=row,
        col=col,
    )
    # Add one specific tick label on the x axis
    # fig.update_xaxes(tickvals=[Var.get_mean(p)], row=row, col=col)
    # Add distribution's mean as an annotation with small font size
    if annot:
        fig.add_annotation(
            x=Var.get_mean(p),
            y=max(p) * 1.1,
            text=f"{Var.get_mean(p):.2f}",
            showarrow=False,
            font=dict(size=8),
            row=row,
            col=col,
        )
    if clean_ticks and Var.name == "Airway resistance (%)":
        fig.update_xaxes(tickvals=np.linspace(Var.a, Var.b, 10), row=row, col=col)
    return -1


def plot_histogram_discrete(
    fig,
    Var: mh.VariableNode,
    p,
    xmin,
    xmax,
    row,
    col,
    title=None,
    colour=None,
    annot=True,
):
    fig.add_trace(
        go.Bar(
            x=Var.midbins,
            y=p,
            # histfunc="sum",  # Use 'sum' to represent pre-counted data
            # xbins=dict(start=xmin, end=xmax, size=Var.bin_width),
        ),
        row=row,
        col=col,
    )
    if colour:
        fig.update_traces(marker_color=colour, row=row, col=col)

    fig.update_xaxes(
        range=[xmin, xmax],
        nticks=20,
        title=title,
        row=row,
        col=col,
    )
    # Add one specific tick label on the x axis
    # fig.update_xaxes(tickvals=[Var.get_mean(p)], row=row, col=col)
    # Add distribution's mean as an annotation with small font size
    if annot:
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
