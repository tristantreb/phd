import time
from functools import reduce
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import src.data.helpers as dh
import src.inference.helpers as ih
import src.models.helpers as mh

plotsdir = dh.get_path_to_main() + "/PlotsBreathe/Longitudinal_model/"


class SharedNodeVariable:
    def __init__(self, name, card, graph_key):
        self.name = name
        self.card = card
        self.graph_key = graph_key
        self.messages = {}
        self.aggregated_message_up = np.ones(card)
        self.posteriors = []

    def add_message(self, plate_key, message):
        assert message.shape == (
            self.card,
        ), "The message must have the same shape as the variable's cardinality"
        # Always replace the message, even if it already exists
        self.messages[plate_key] = message
        agg_m = np.multiply(self.aggregated_message_up, message)
        # Renormalise
        self.aggregated_message_up = agg_m / agg_m.sum()

    def get_virtual_message(self, plate_key):
        # Use the aggregated message instead
        agg_m = self.aggregated_message_up
        # Remove previous today's message from agg_m
        if plate_key in self.messages.keys():
            curr_m = self.messages[plate_key]
            agg_m = np.divide(
                agg_m, curr_m, out=np.zeros_like(agg_m), where=curr_m != 0
            )
            agg_m = agg_m / agg_m.sum()
        return agg_m

        # Multiply all messages together (less efficient)
        # Remove message with plate_key from the list of messages
        # messages_up = self.messages.copy()
        # if plate_key in self.messages.keys():
        #     messages_up.pop(plate_key)

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


def query_across_days(
    df,
    belief_propagation,
    shared_variables: List[SharedNodeVariable],
    variables: List[str],
    evidence_variables: List[str],
    diff_threshold,
):
    final_epoch = False
    epoch = 0
    df_res_shared = pd.DataFrame(
        columns=["Epoch"] + list(map(lambda v: v.name, shared_variables))
    )

    df_res_vars = pd.DataFrame(columns=["Day"] + list(map(lambda v: v.name, variables)))

    # Initialize posteriors distribution to uniform
    posteriors_old = [
        get_uniform_message(shared_var.card) for shared_var in shared_variables
    ]

    while True:
        print(f"epoch {epoch}")

        for i in range(len(df)):
            day = df["Date Recorded"].iloc[i].strftime("%Y-%m-%d")

            def build_evidence(variables):
                evidence = {}
                for variable in variables:
                    idx_obs = df[variable].iloc[i]
                    evidence[variable] = idx_obs
                return evidence

            evidence = build_evidence(evidence_variables)

            def build_virtual_evidence(shared_variables):
                virtual_evidence = {}
                for shared_var in shared_variables:
                    # tic = time.time()
                    virtual_message = shared_var.get_virtual_message(day)
                    # toc = time.time()
                    # print(
                    #     f"Time to get virtual message for {shared_var.name}: {toc-tic}"
                    # )
                    if virtual_message is not None:
                        virtual_evidence[shared_var.name] = virtual_message
                return virtual_evidence

            virtual_evidence = build_virtual_evidence(shared_variables)

            # Query the graph
            if final_epoch:
                vars_to_infer = shared_variables + variables
                vars_to_infer = list(map(lambda v: v.name, vars_to_infer))
                res = belief_propagation.query(
                    vars_to_infer, evidence, virtual_evidence, get_messages=False
                )
                new_row = [day] + list(map(lambda v: res[v].values, vars_to_infer))
                new_row = pd.DataFrame([new_row], columns=["Day"] + vars_to_infer)
                df_res_vars = pd.concat([df_res_vars, new_row], ignore_index=True)

            else:
                vars_to_infer = list(map(lambda v: v.name, shared_variables))
                res, messages = belief_propagation.query(
                    vars_to_infer, evidence, virtual_evidence, get_messages=True
                )
                # Save message for current day
                for shared_var in shared_variables:
                    shared_var.add_message(day, messages[shared_var.graph_key])

        posteriors_old, diffs = get_diffs(res, posteriors_old, shared_variables)

        for shared_var, diff in zip(shared_variables, diffs):
            print(f"Epoch {epoch} - Posteriors' diff for {shared_var.name}: {diff}")

        # Create new row df with epoch, and on shared variables array per row cel
        new_row = [epoch] + list(map(lambda v: res[v.name].values, shared_variables))
        # Same but as df
        new_row = pd.DataFrame(
            [new_row], columns=["Epoch"] + list(map(lambda v: v.name, shared_variables))
        )

        df_res_shared = pd.concat([df_res_shared, new_row], ignore_index=True)

        if np.sum(diffs) < diff_threshold:
            if final_epoch:
                # Terminates the query
                return df_res_vars, df_res_shared
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


def get_heatmap_data(df, var: mh.variableNode):
    data = (
        np.array([item for sublist in df[var.name] for item in sublist])
        .reshape(len(df), len(var.bins))
        .T
    )
    return pd.DataFrame(index=var.bins_str, columns=df.Day, data=data)


def plot_heatmap(fig, df, var, row, col, coloraxis):
    x = df.columns
    y = df.index
    z = df

    fig.add_trace(go.Heatmap(z=z, x=x, y=y, coloraxis=coloraxis), row=row, col=col)
    # Update yaxis properties
    fig.update_yaxes(title_text=var.name, row=row, col=col)
    # fig.update_xaxes(title_text="Days", row=row, col=col)


def plot_posterior_validation(
    df_query_res,
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

    df_res_hfev1 = get_heatmap_data(df_query_res, HFEV1)
    df_res_ho2sat = get_heatmap_data(df_query_res, HO2Sat)
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
    ecFEV1: mh.variableNode,
    O2Sat: mh.variableNode,
    df_query_res: pd.DataFrame,
    AR: mh.variableNode,
    IA: mh.variableNode,
    HFEV1: mh.variableNode,
    HO2Sat: mh.variableNode,
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
