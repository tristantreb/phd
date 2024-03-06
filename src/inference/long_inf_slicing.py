from functools import reduce
from typing import List

import numpy as np
import pandas as pd


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
):
    epoch = 0
    df_res = pd.DataFrame(
        columns=["Epoch"] + list(map(lambda v: v.name, shared_variables))
    )

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

            evidence = build_evidence(variables)

            def build_virtual_evidence(shared_variables):
                virtual_evidence = {}
                for shared_var in shared_variables:
                    virtual_message = shared_var.get_virtual_message(day)
                    if virtual_message is not None:
                        virtual_evidence[shared_var.name] = virtual_message
                return virtual_evidence

            virtual_evidence = build_virtual_evidence(shared_variables)

            var_to_infer = list(map(lambda v: v.name, shared_variables))

            # Query the graph
            res, messages = belief_propagation.query(
                var_to_infer, evidence, virtual_evidence, get_messages=True
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

        df_res = pd.concat([df_res, new_row], ignore_index=True)

        if np.sum(diffs) == 0:
            print("All diffs are 0, inference converged")
            return df_res
        epoch += 1
