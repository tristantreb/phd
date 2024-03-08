from functools import reduce
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pgmpy.factors.discrete import TabularCPD
from plotly.subplots import make_subplots

import src.data.helpers as dh
import src.inference.helpers as ih
import src.models.helpers as mh

plotsdir = dh.get_path_to_main() + "/PlotsBreathe/Longitudinal_model/"


class SharedVariableNode:
    def __init__(self, name, card, factor_node_key):
        self.name = name
        self.card = card
        self.factor_node_key = factor_node_key
        self.virtual_messages = {}
        self.agg_virtual_message = np.ones(card)

    def add_or_update_message(self, day_key, new_message):
        assert new_message.shape == (
            self.card,
        ), "The message must have the same shape as the variable's cardinality"
        # Always replace the message for that day, even if it already exists
        self.virtual_messages[day_key] = new_message

    def set_agg_virtual_message(self, virtual_message, new_message):
        """
        The new aggregated message is the multiplication of all messages coming from the factor to the node

        Virtual message: multiplication of all factor to node messages excluding current day message
        New message: newly computed factor to node message
        """
        agg_m = np.multiply(virtual_message, new_message)
        self.agg_virtual_message = agg_m / agg_m.sum()

    def get_virtual_message(self, day_key):
        """
        Returns the aggregated message, excluding the message from the current day
        if applicable (if n_epoch > 0).
        """
        agg_m = self.agg_virtual_message

        if day_key not in self.virtual_messages.keys():
            return agg_m

        # Remove previous today's message from agg_m
        curr_m = self.virtual_messages[day_key]
        agg_m_excl_curr_m = np.divide(
            agg_m, curr_m, out=np.zeros_like(agg_m), where=curr_m != 0
        )
        return agg_m_excl_curr_m / agg_m_excl_curr_m.sum()

        # Multiply all messages together (less efficient)
        # Remove message with day_key from the list of messages
        # messages_up = self.messages.copy()
        # if day_key in self.messages.keys():
        #     messages_up.pop(day_key)

        # if len(messages_up) == 0:
        #     return None
        # elif len(messages_up) == 1:
        #     return list(messages_up.values())[0]
        # else:
        #     message_up = np.ones(self.card)
        #     for message in messages_up.values():
        #         message_up = np.multiply(message_up, message)
        #         # Renormalise each time to avoid numerical issues (message going to 0)
        #         message_up = message_up / message_up.sum()
        #     return message_up


def get_diffs(res, posteriors_old, vars):
    diffs = []
    posteriors_new = []
    for var, old in zip(vars, posteriors_old):
        new, diff = get_diff(res, old, var)
        diffs.append(diff)
        posteriors_new.append(new)
    return posteriors_new, diffs


def get_diff(res, old, var):
    new = res[var.name].values
    diff = np.abs(new - old).sum()
    return new, diff


def get_uniform_message(card):
    return np.ones(card) / card


def get_var_name_list(variables: List[mh.VariableNode] | List[SharedVariableNode]):
    return list(map(lambda v: v.name, variables))


def save_res_to_df(df, day, res, variables):
    new_row = [day] + list(map(lambda v: res[v].values, variables))
    new_row = pd.DataFrame([new_row], columns=["Day"] + variables)
    return pd.concat([df, new_row], ignore_index=True)


def build_evidence(df, i, variables):
    evidence = {}
    for variable in variables:
        idx_obs = df[variable].iloc[i]
        evidence[variable] = idx_obs
    return evidence


def build_virtual_evidence(shared_variables, day):
    virtual_evidence = []
    virtual_messages = {}
    for shared_var in shared_variables:
        virtual_message = shared_var.get_virtual_message(day)
        if virtual_message is not None:
            virtual_evidence.append(
                TabularCPD(
                    shared_var.name,
                    shared_var.card,
                    virtual_message.reshape(-1, 1),
                )
            )
            virtual_messages[shared_var.name] = virtual_message
    return virtual_evidence, virtual_messages


def query_across_days(
    df,
    belief_propagation,
    shared_variables: List[SharedVariableNode],
    variables: List[str],
    evidence_variables: List[str],
    diff_threshold,
):
    final_epoch = False
    epoch = 0

    df_res_before_convergence = pd.DataFrame(
        columns=["Day"] + list(map(lambda v: v.name, shared_variables))
    )
    df_res_final_epoch = pd.DataFrame(
        columns=["Day"] + list(map(lambda v: v.name, variables))
    )

    # Initialize posteriors distribution to uniform
    posteriors_old = [
        get_uniform_message(shared_var.card) for shared_var in shared_variables
    ]

    while True:
        print(f"epoch {epoch}")

        for i in range(len(df)):
            day = df["Date Recorded"].iloc[i].strftime("%Y-%m-%d")
            # Get query inputs
            evidence = build_evidence(df, i, evidence_variables)
            virtual_evidence, virtual_messages = build_virtual_evidence(
                shared_variables, day
            )

            if final_epoch:
                # Query all variables to get all posteriors
                vars_to_infer = get_var_name_list(shared_variables + variables)
                query_res = belief_propagation.query(
                    vars_to_infer, evidence, virtual_evidence
                )
                df_res_final_epoch = save_res_to_df(
                    df_res_final_epoch,
                    day,
                    query_res,
                    vars_to_infer,
                )
            else:
                # Query shared variables to get cross plate message
                vars_to_infer = get_var_name_list(shared_variables)
                query_res, query_messages = belief_propagation.query(
                    vars_to_infer, evidence, virtual_evidence, get_messages=True
                )
                df_res_before_convergence = save_res_to_df(
                    df_res_before_convergence,
                    f"{epoch}, {day}",
                    query_res,
                    vars_to_infer,
                )
                for shared_var in shared_variables:
                    # Get newly computed message from the query output
                    new_message = query_messages[shared_var.factor_node_key]
                    shared_var.add_or_update_message(day, new_message)
                    shared_var.set_agg_virtual_message(
                        virtual_messages[shared_var.name], new_message
                    )

        posteriors_old, diffs = get_diffs(query_res, posteriors_old, shared_variables)

        for shared_var, diff in zip(shared_variables, diffs):
            print(f"Epoch {epoch} - Posteriors' diff for {shared_var.name}: {diff}")

        if np.sum(diffs) < diff_threshold or epoch > 99:
            if final_epoch:
                return df_res_final_epoch, df_res_before_convergence
            print(
                f"All diffs are below {diff_threshold}, running another epoch to get all posteriors"
            )
            final_epoch = True
        epoch += 1


def plot_scatter(fig, x, y, row, col, colour=None, title=None):
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
        ),
        row=row,
        col=col,
    )
    fig.update_traces(marker=dict(size=2), row=row, col=col)
    if colour:
        fig.update_traces(marker=dict(color=colour), row=row, col=col)
    # Add x axis title
    fig.update_yaxes(title_text=title, row=row, col=col)
    # fig.update_xaxes(title_text="Days", row=row, col=col)


def get_heatmap_data(df, var: mh.VariableNode):
    data = (
        np.array([item for sublist in df[var.name] for item in sublist])
        .reshape(len(df), var.card)
        .T
    )
    return pd.DataFrame(index=var.get_bins_str(), columns=df.Day, data=data)


def plot_heatmap(fig, df, var, row, col, coloraxis):
    x = df.columns
    y = df.index
    z = df

    fig.add_trace(go.Heatmap(z=z, x=x, y=y, coloraxis=coloraxis), row=row, col=col)
    # Update yaxis properties
    fig.update_yaxes(title_text=var.name, row=row, col=col)
    # fig.update_xaxes(title_text="Days", row=row, col=col)


def plot_posterior_validation(
    df_res_before_convergence,
    HFEV1shared,
    HFEV1,
    HO2Satshared,
    HO2Sat,
    df_breathe,
    ecFEV1,
    O2Sat,
    colorscale=[[0, "lightcyan"], [0.5, "yellow"], [1, "blue"]],
    save=False,
):
    layout = [
        [{"type": "scatter", "rowspan": 1}],
        [{"type": "scatter", "rowspan": 1}],
        [{"type": "heatmap", "rowspan": 2}],
        [None],
        [{"type": "heatmap", "rowspan": 2}],
        [None],
    ]
    fig = make_subplots(
        rows=np.shape(layout)[0],
        cols=np.shape(layout)[1],
        specs=layout,
        vertical_spacing=0.01,
        shared_xaxes=True,
    )
    plot_scatter(
        fig,
        df_breathe["Date Recorded"],
        df_breathe["ecFEV1"],
        row=1,
        col=1,
        colour="black",
        title=ecFEV1.name,
    )
    plot_scatter(
        fig,
        df_breathe["Date Recorded"],
        df_breathe["O2 Saturation"],
        row=2,
        col=1,
        colour="black",
        title=O2Sat.name,
    )

    df_res_hfev1 = get_heatmap_data(df_res_before_convergence, HFEV1)
    df_res_ho2sat = get_heatmap_data(df_res_before_convergence, HO2Sat)
    plot_heatmap(fig, df_res_hfev1, HFEV1shared, row=3, col=1, coloraxis="coloraxis1")
    plot_heatmap(fig, df_res_ho2sat, HO2Satshared, row=5, col=1, coloraxis="coloraxis2")

    title = f"ID {df_breathe.ID[0]} - Longitudinal inference stability validation"
    fig.update_layout(
        title=title,
        width=1200,
        height=800,
        font=dict(size=5),
        showlegend=False,
        coloraxis1=dict(
            colorscale=colorscale,
            colorbar_x=1,
            colorbar_y=0.5,
            colorbar_thickness=23,
            colorbar_len=0.3,
        ),
        coloraxis2=dict(
            colorscale=colorscale,
            colorbar_x=1,
            colorbar_y=0.15,
            colorbar_thickness=23,
            colorbar_len=0.3,
        ),
    )
    if save:
        fig.write_image(
            f"{plotsdir}Healthy_vars_inference_stability_validation/{title}.pdf"
        )
    else:
        fig.show()


def plot_query_res(
    df_breathe: pd.DataFrame,
    ecFEV1: mh.VariableNode,
    O2Sat: mh.VariableNode,
    df_query_res: pd.DataFrame,
    AR: mh.VariableNode,
    IA: mh.VariableNode,
    HFEV1: mh.VariableNode,
    HO2Sat: mh.VariableNode,
    colorscale=[[0, "lightcyan"], [0.5, "yellow"], [1, "blue"]],
    save=False,
):
    layout = [
        [{"type": "scatter", "rowspan": 1}, {"type": "scatter", "rowspan": 1}],
        [{"type": "scatter", "rowspan": 1}, {"type": "scatter", "rowspan": 1}],
        [{"type": "heatmap", "rowspan": 2}, {"type": "heatmap", "rowspan": 2}],
        [None, None],
        [{"type": "scatter", "rowspan": 1}, {"type": "scatter", "rowspan": 1}],
    ]
    fig = make_subplots(
        rows=np.shape(layout)[0],
        cols=np.shape(layout)[1],
        specs=layout,
        vertical_spacing=0.05,
    )
    # Priors
    ih.plot_histogram(
        fig, HFEV1, HFEV1.cpt, HFEV1.a, HFEV1.b, row=1, col=1, colour="#636EFA"
    )
    ih.plot_histogram(
        fig, HO2Sat, HO2Sat.cpt, HO2Sat.a, HO2Sat.b, row=1, col=2, colour="#636EFA"
    )

    # Posteriors for shared variables
    hfev1_posterior = df_query_res[HFEV1.name].iloc[-1]
    ih.plot_histogram(
        fig, HFEV1, hfev1_posterior, HFEV1.a, HFEV1.b, row=2, col=1, colour="#636EFA"
    )
    ho2sat_posterior = df_query_res[HO2Sat.name].iloc[-1]
    ih.plot_histogram(
        fig,
        HO2Sat,
        ho2sat_posterior,
        HO2Sat.a,
        HO2Sat.b,
        row=2,
        col=2,
        colour="#636EFA",
    )

    # Heatmaps
    df_query_res_ar = get_heatmap_data(df_query_res, AR)
    df_query_res_ia = get_heatmap_data(df_query_res, IA)
    plot_heatmap(fig, df_query_res_ar, AR, row=3, col=1, coloraxis="coloraxis1")
    plot_heatmap(fig, df_query_res_ia, IA, row=3, col=2, coloraxis="coloraxis2")

    # Observations
    plot_scatter(
        fig,
        df_breathe["Date Recorded"],
        df_breathe["ecFEV1"],
        row=5,
        col=1,
        colour="black",
        title=ecFEV1.name,
    )
    plot_scatter(
        fig,
        df_breathe["Date Recorded"],
        df_breathe["O2 Saturation"],
        row=5,
        col=2,
        colour="black",
        title=O2Sat.name,
    )

    title = f"ID {df_breathe.ID[0]} - Longitudinal inference results"
    fig.update_layout(
        title=title,
        height=900,
        width=1300,
        font=dict(size=5),
        showlegend=False,
        coloraxis1=dict(
            colorscale=colorscale,
            colorbar_x=0.45,
            colorbar_y=0.402,
            colorbar_thickness=23,
            colorbar_len=0.415,
            colorbar={"title": "AR"},
        ),
        coloraxis2=dict(
            colorscale=colorscale,
            colorbar_x=1,
            colorbar_y=0.402,
            colorbar_thickness=23,
            colorbar_len=0.415,
            colorbar={"title": "IA"},
        ),
    )
    if save:
        fig.write_image(f"{plotsdir}Healthy_vars_inference_results/{title}.pdf")
    else:
        fig.show()
