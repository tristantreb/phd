import numpy as np
import pandas as pd
import plotly.graph_objects as go

import src.models.builders as mb
import src.models.graph_builders as graph_builders
import src.models.helpers as mh
import src.models.var_builders as var_builders
from pgmpy.inference import BeliefPropagation, BeliefPropagationWithMessageParsing
from src.inference.inf_algs import apply_bayes_net_bp, apply_factor_graph_bp

# Set global value for tolerance.
# This to account for the rounding error: https://www.cs.drexel.edu/~jpopyack/Courses/CSP/Fa17/extras/Rounding/index.html#:~:text=Rounding%20(roundoff)%20error%20is%20a,word%20size%20used%20for%20integers.
TOL_GLOBAL = 1e-6
# Switch from 1e-8 to 1e-6 to because got 0.9999999885510139 sum of probabilities for a model with AW


def infer_on_factor_graph(
    inference_alg: BeliefPropagationWithMessageParsing,
    variables: tuple[mh.VariableNode],
    evidences: tuple[tuple[mh.VariableNode, float]],
    virtual_evidence=None,
    get_messages=False,
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

    res = inference_alg.query(
        variables=list(map(lambda v: v.name, variables)),
        evidence=evidences_binned,
        virtual_evidence=virtual_evidence,
        get_messages=get_messages,
    )
    return res


def infer(
    inference_alg: BeliefPropagation,
    variables: tuple[mh.VariableNode],
    evidences: tuple[tuple[mh.VariableNode, float]],
    show_progress=False,
    joint=False,
    debug=False,
):
    """
    Runs an inference query against a given PGMPY inference model, variables, evidences
    :param inference_alg: The inference algorithm to use
    :param variables: The variables to query
    :param evidences: The evidences to use

    :return: The result of the inference
    """
    variables = list(map(lambda v: v.name, variables))
    if debug:
        print(f"Variables to infer: {variables}")

    evidences_binned = dict()
    for [evidence_var, value] in evidences:
        [_bin, bin_idx] = get_bin_for_value(value, evidence_var)
        evidences_binned.update({evidence_var.name: bin_idx})
        if debug:
            print(f"Evidence for {evidence_var.name}: value {value}, idx {bin_idx}")

    res = inference_alg.query(
        variables=variables,
        evidence=evidences_binned,
        show_progress=show_progress,
        joint=joint,
    )
    return res


# Given an observation and an array of bins, this returns the bin that the value falls into
def get_bin_for_value(obs: float, var: mh.VariableNode, tol=TOL_GLOBAL):
    """
    Obsolete as this function has been added ot the VariableNode class
    """
    # Center bins around value observed
    relative_bins = var.bins - obs - tol

    # Find the highest negative value of the bins relative to centered bins
    idx = np.where(relative_bins <= 0, relative_bins, -np.inf).argmax()

    (lower_idx, upper_idx) = var.get_bins_arr()[idx]
    return "[{}; {})".format(lower_idx, upper_idx), idx


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


def infer_vars_and_get_back_df(
    df,
    variables_to_infer,
    observed_variables,
    ecFEF2575prctecFEV1_cpt=None,
    IA_cpt=None,
):
    """
    Infer AR, IA, HFEV1, HO2Sat fo each entry in the dataset, for the given observed variables as evidence
    """

    def infer_vars_for_ID(df):
        df.reset_index(inplace=True)

        (
            HFEV1,
            ecFEV1,
            AR,
            HO2Sat,
            O2SatFFA,
            IA,
            UO2Sat,
            O2Sat,
            ecFEF2575prctecFEV1,
        ) = var_builders.o2sat_fev1_fef2575_point_in_time_model_shared_healthy_vars(
            df.Height[0], df.Age[0], df.Sex[0]
        )

        # Update cpt to custom one if provided
        if ecFEF2575prctecFEV1_cpt is not None:
            ecFEF2575prctecFEV1.set_cpt(ecFEF2575prctecFEV1_cpt)
        if IA_cpt is None:
            model = graph_builders.fev1_fef2575_o2sat_point_in_time_factor_graph(
                HFEV1,
                ecFEV1,
                AR,
                HO2Sat,
                O2SatFFA,
                IA,
                UO2Sat,
                O2Sat,
                ecFEF2575prctecFEV1,
                False,
            )
            inf_alg = apply_factor_graph_bp(model)
        else:
            # Else we model AR causing IA with the given CPT
            IA.set_cpt(IA_cpt)
            # Since this introduces a loop we have to use a Bayes Net to run the inference
            model = graph_builders.fev1_o2sat_fef2575_point_in_time_model(
                HFEV1,
                ecFEV1,
                AR,
                HO2Sat,
                O2SatFFA,
                IA,
                UO2Sat,
                O2Sat,
                ecFEF2575prctecFEV1,
                False,
            )
            inf_alg = apply_bayes_net_bp(model)

        def infer_and_unpack(row):
            # Build evidence
            evidence = [
                [obs_var, row[obs_var.get_colname()]] for obs_var in observed_variables
            ]

            if IA_cpt is None:
                res = infer_on_factor_graph(
                    inf_alg,
                    variables_to_infer,
                    evidence,
                )
            else:
                # Infer on Bayes net
                res = infer(
                    inf_alg,
                    variables_to_infer,
                    evidence,
                )

            res_values = (res[var.name].values for var in variables_to_infer)

            return row["Date Recorded"], *res_values

        res = df.apply(infer_and_unpack, axis=1)
        return res

    variables_to_infer_dict = {
        i + 1: variables_to_infer[i].get_abbr() for i in range(len(variables_to_infer))
    }
    variables_to_infer_dict[0] = "Date Recorded"

    resraw = df.groupby("ID").apply(infer_vars_for_ID)
    # resraw = df.iloc[np.r_[10:13, 3000:3007]].groupby("ID").apply(infer_vars_for_ID)
    res = (
        resraw.apply(pd.Series)
        .reset_index()
        .rename(columns=variables_to_infer_dict)
        .drop(columns="level_1")
    )

    for var in variables_to_infer:
        res[f"{var.get_abbr()} mean"] = res[var.get_abbr()].apply(
            lambda x: var.get_mean(x)
        )

    return res
