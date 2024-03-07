from functools import reduce
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import src.inference.helpers as ih


class SharedNodeVariable:
    def __init__(self, name, card, graph_key):
        self.name = name
        self.card = card
        self.graph_key = graph_key
        self.messages = {}
        self.posteriors = []

    def add_message(self, plate_key, message):
        assert message.shape == (
            self.card,
        ), "The message must have the same shape as the variable's cardinality"
        # Always replace the message, even if it already exists
        self.messages[plate_key] = message

    def get_virtual_message(self, plate_key):
        # Remove message with plate_key from the list of messages
        messages_up = self.messages.copy()
        if plate_key in self.messages.keys():
            messages_up.pop(plate_key)
        if len(messages_up) == 0:
            return None
        elif len(messages_up) == 1:
            return list(messages_up.values())[0]
        else:
            return reduce(np.multiply, list(messages_up.values()))


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
                    virtual_message = shared_var.get_virtual_message(day)
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

        if np.sum(diffs) == 0:
            if final_epoch:
                # Terminates the query
                return df_res_vars, df_res_shared
            print("All diffs are 0, rerunning a last inference to get all posteriors")
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
    fig.update_xaxes(title_text="Days", row=row, col=col)


def plot_heatmap(fig, df, shared_var, row, col, coloraxis):
    x = df.columns
    y = df.index
    z = df

    fig.add_trace(go.Heatmap(z=z, x=x, y=y, coloraxis=coloraxis), row=row, col=col)
    fig.update_layout(
        yaxis_title=shared_var.name,
        xaxis_title="Day",
    )


def plot_res_for_ID_just_shared_variables(
    df, HFEV1, HO2Sat, ecFEV1, O2Sat, hfev1_posterior, ho2sat_posterior
):
    fig = go.Figure()
    fig = make_subplots(rows=4, cols=1)
    ih.plot_histogram(fig, HFEV1, hfev1_posterior, HFEV1.a, HFEV1.b, 1, 1, "#636EFA")
    ih.plot_histogram(
        fig, HO2Sat, ho2sat_posterior, HO2Sat.a, HO2Sat.b, 2, 1, "#636EFA"
    )

    plot_scatter(
        fig,
        df["Date Recorded"],
        df[ecFEV1.name],
        3,
        1,
        "black",
        ecFEV1.name,
    )
    plot_scatter(
        fig,
        df["Date Recorded"],
        df[O2Sat.name],
        4,
        1,
        "black",
        O2Sat.name,
    )
    fig.update_layout(
        title=f"ID 101 - HFEV1 and HO2Sat posterior distributions over time ({len(df)} data-points)",
        width=1000,
        height=800,
        font=dict(size=9),
        showlegend=False,
    )
    fig.show()
