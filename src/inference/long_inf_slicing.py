from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import src.data.helpers as dh
import src.inference.helpers as ih
import src.models.cpts.helpers as cpth
import src.models.helpers as mh
from pgmpy.factors.discrete import TabularCPD

plotsdir = dh.get_path_to_main() + "/PlotsBreathe/Longitudinal_model/"


def get_diffs(res, posteriors_old, vars):
    """
    Returns the sum of the absolute elementwise difference between the new and old posteriors
    """

    def get_diff(res, old, var):
        new = res[var.name].values
        diff = np.abs(new - old).sum()
        return new, diff

    diffs = []
    posteriors_new = []
    for var, old in zip(vars, posteriors_old):
        new, diff = get_diff(res, old, var)
        diffs.append(diff)
        posteriors_new.append(new)
    return posteriors_new, diffs


def get_uniform_message(card):
    return np.ones(card) / card


def get_var_name_list(variables: List[mh.VariableNode] | List[mh.SharedVariableNode]):
    return list(map(lambda v: v.name, variables))


def save_res_to_df(df, day, res, variables_to_infer, row, evidence_vars):
    evidence_vars = list(
        map(lambda e: mh.abbr_to_colname(mh.name_to_abbr(e)), evidence_vars)
    )
    evidence_values = list(row[evidence_vars])
    inferred_dist = list(map(lambda v: res[v].values, variables_to_infer))
    new_row = pd.DataFrame(
        data=[
            [row["ID"], day, row["Age"], row["Height"], row["Sex"]]
            + evidence_values
            + inferred_dist
        ],
        columns=["ID", "Day", "Age", "Height", "Sex"]
        + evidence_vars
        + variables_to_infer,
    )
    return pd.concat([df, new_row], ignore_index=True)


def build_evidence(df, i, variables):
    evidence = {}
    for variable in variables:
        idx_obs = df["idx " + variable].iloc[i]
        evidence[variable] = idx_obs
    return evidence


def build_virtual_evidence_shared_vars(shared_variables, day):
    vevidence = []
    for shared_var in shared_variables:
        virtual_message = shared_var.get_virtual_message(day, agg_method=True)
        if virtual_message is not None:
            vevidence.append(
                TabularCPD(
                    shared_var.name,
                    shared_var.card,
                    virtual_message.reshape(-1, 1),
                )
            )
    return vevidence


def build_vevidence_ar(AR, day, prev_day_key=None, next_day_key=None):
    """
    Builds a virtual evidence for the airway resistance variable given the previous and next days
    """
    prior = AR.get_virtual_message(day, prev_day_key, next_day_key)
    return TabularCPD(AR.name, AR.card, prior.reshape(-1, 1))


def query_forwardly_across_days(
    df,
    belief_propagation,
    shared_variables: List[mh.SharedVariableNode],
    variables: List[mh.VariableNode],
    evidence_variables: List[str],
    diff_threshold,
    debug=True,
    auto_reset_shared_vars=True,
):
    """
    algorithm to query the point in time model forwardly across days, thus making an approximate longitudinal inference

    variables: contains the names of the variables to infer, as defined in the graph
    evidence_variables: contains names of the observed variables, as defined in the df's columns
    auto_reset_shared_vars: bool to automatically reset the shared variables after the computations are done
    """
    df = df.reset_index(drop=True)
    final_epoch = False
    epoch = 0

    # Check that each date in Date Redorded is unique
    assert df["Date Recorded"].nunique() == len(
        df
    ), "Error: Cannot process input df as there are doublons in the Date Recorded column."

    df_res_before_convergence = pd.DataFrame({})
    df_res_final_epoch = pd.DataFrame({})

    # Initialize posteriors distribution to uniform
    posteriors_old = [
        get_uniform_message(shared_var.card) for shared_var in shared_variables
    ]

    while True:
        if debug:
            print(f"epoch {epoch}")

        for i, row in df.iterrows():
            day = row["Date Recorded"].strftime("%Y-%m-%d")

            if debug:
                for shared_var in shared_variables:
                    shared_var.reset()
            # Get query inputs
            evidence_dict = build_evidence(df, i, evidence_variables)
            vevidence = build_virtual_evidence_shared_vars(shared_variables, day)

            vars_to_infer = get_var_name_list(shared_variables + variables)
            if final_epoch:
                # Query all variables to get all posteriors
                if debug:
                    vevidence_str = [(cpd.variable, cpd.values) for cpd in vevidence]
                    print(
                        f"Querying all variables: {vars_to_infer} with evidence: {evidence_dict} and virtual evidence: {vevidence_str}"
                    )
                query_res = belief_propagation.query(
                    vars_to_infer, evidence_dict, vevidence
                )
                df_res_final_epoch = save_res_to_df(
                    df_res_final_epoch,
                    day,
                    query_res,
                    vars_to_infer,
                    row,
                    evidence_variables,
                )
            else:
                # Query shared variables to get cross plate message
                if debug:
                    vevidence_str = [(cpd.variable, cpd.values) for cpd in vevidence]
                    print(
                        f"Querying all variables: {vars_to_infer} with evidence: {evidence_dict} and virtual evidence: {vevidence_str}"
                    )
                query_res, query_messages = belief_propagation.query(
                    vars_to_infer, evidence_dict, vevidence, get_messages=True
                )
                df_res_before_convergence = save_res_to_df(
                    df_res_before_convergence,
                    f"{epoch}, {day}",
                    query_res,
                    vars_to_infer,
                    row,
                    evidence_variables,
                )
                for shared_var in shared_variables:
                    # Get newly computed message from the query output
                    new_message = query_messages[shared_var.factor_node_key]
                    shared_var.add_or_update_message(day, new_message)
                    if len(vevidence) == 0:
                        vmessage = get_uniform_message(shared_var.card)
                    else:
                        vmessage = [
                            cpd.values
                            for cpd in vevidence
                            if cpd.variable == shared_var.name
                        ][0]
                    shared_var.set_agg_virtual_message(vmessage, new_message)
                    shared_var.set_agg_virtual_message(vmessage, new_message)

        posteriors_old, diffs = get_diffs(query_res, posteriors_old, shared_variables)

        for shared_var, diff in zip(shared_variables, diffs):
            if debug:
                print(f"Epoch {epoch} - Posteriors' diff for {shared_var.name}: {diff}")

        # Convergence reached when the diff is below the threshold
        # or when the maximum number of epochs is reached
        # When convergence is reached, run another epoch to get all posteriors
        if np.sum(diffs) < diff_threshold or epoch > 99:
            if final_epoch:
                if auto_reset_shared_vars:
                    for shared_var in shared_variables:
                        shared_var.reset()
                return df_res_final_epoch, df_res_before_convergence, shared_variables
            if debug:
                print(
                    f"All diffs are below {diff_threshold}, running another epoch to get all posteriors"
                )
            final_epoch = True
        epoch += 1


def query_back_and_forth_across_days(
    df_init,
    belief_propagation,
    shared_variables: List[mh.SharedVariableNode],
    variables: List[mh.VariableNode],
    evidence_variables: List[str],
    diff_threshold,
    debug=False,
    auto_reset_shared_vars=True,
    max_passes=99,
):
    """
    algorithm to query the point in time model across days, thus making an approximate longitudinal inference
    the algorithm atlernates forward and backward passes, as a first step towards back building an algorithm that can handle inter-day dependencies

    variables: contains the names of the variables to infer, as defined in the graph
    evidence_variables: contains names of the observed variables, as defined in the df's columns
    auto_reset_shared_vars: bool to automatically reset the shared variables after the computations are done
    """
    df_init = df_init.reset_index(drop=True)
    final_pass = False
    passes = 1
    previous_date = None

    # Check that each date in Date Redorded is unique
    assert df_init["Date Recorded"].nunique() == len(
        df_init
    ), "Error: Cannot process input df as there are doublons in the Date Recorded column."

    df_res_before_convergence = pd.DataFrame({})
    df_res_final_epoch = pd.DataFrame({})

    # Initialize posteriors distribution to uniform
    posteriors_old = [
        get_uniform_message(shared_var.card) for shared_var in shared_variables
    ]

    while True:
        if final_pass:
            if debug:
                print(f"Final pass - pass {passes} (forward)")
            df = df_init.sort_values("Date Recorded", ascending=True).reset_index(
                drop=True
            )
            previous_date = None
        else:
            forward_pass = passes % 2 == 1
            if forward_pass:
                if debug:
                    print(f"Pass {passes} (forward)")
                df = df_init.sort_values("Date Recorded", ascending=True).reset_index(
                    drop=True
                )
            else:
                if debug:
                    print(f"Pass {passes} (backward)")
                df = df_init.sort_values("Date Recorded", ascending=False).reset_index(
                    drop=True
                )
            if passes != 1:
                # Remove the first element as it's the same as the last element of the previous pass
                df = df.iloc[1:].reset_index(drop=True)

        for i, row in df.iterrows():
            date = row["Date Recorded"]
            date_str = date.strftime("%Y-%m-%d")

            assert (
                previous_date != date
            ), "The dates are the same, no inference can be made"

            # Get query inputs
            evidence_dict = build_evidence(df, i, evidence_variables)
            vevidence_shared = build_virtual_evidence_shared_vars(
                shared_variables, date_str
            )
            vevidence = vevidence_shared

            vars_to_infer = get_var_name_list(shared_variables + variables)

            if final_pass:
                # Query all variables to get all posteriors
                if debug:
                    vevidence_str = [(cpd.variable, cpd.values) for cpd in vevidence]
                    print(
                        f"Querying all variables: {vars_to_infer} with evidence: {evidence_dict} and virtual evidence: {vevidence_str}"
                    )
                query_res = belief_propagation.query(
                    vars_to_infer, evidence_dict, vevidence
                )
                df_res_final_epoch = save_res_to_df(
                    df_res_final_epoch,
                    date_str,
                    query_res,
                    vars_to_infer,
                    row,
                    evidence_variables,
                )

            else:
                # Query shared variables to get cross plate message
                if debug:
                    vevidence_str = [(cpd.variable, cpd.values) for cpd in vevidence]
                    print(
                        f"Querying all variables: {vars_to_infer} with evidence: {evidence_dict} and virtual evidence: {vevidence_str}"
                    )
                query_res, query_messages = belief_propagation.query(
                    vars_to_infer,
                    evidence_dict,
                    vevidence,
                    get_messages=True,
                )
                df_res_before_convergence = save_res_to_df(
                    df_res_before_convergence,
                    f"{passes}, {date_str}",
                    query_res,
                    vars_to_infer,
                    row,
                    evidence_variables,
                )

                # Get newly computed message from the query output
                for shared_var in shared_variables:
                    new_message = query_messages[shared_var.factor_node_key]
                    shared_var.add_or_update_message(date_str, new_message)
                    if len(vevidence_shared) == 0:
                        vmessage = get_uniform_message(shared_var.card)
                    else:
                        vmessage = [
                            cpd.values
                            for cpd in vevidence_shared
                            if cpd.variable == shared_var.name
                        ][0]
                    shared_var.set_agg_virtual_message(vmessage, new_message)

            previous_date = date

        posteriors_old, diffs = get_diffs(query_res, posteriors_old, shared_variables)

        for shared_var, diff in zip(shared_variables, diffs):
            print(f"Pass {passes} - Posteriors' diff for {shared_var.name}: {diff}")

        # Convergence reached when the diff is below the threshold
        # or when the maximum number of passes is reached
        # When convergence is reached, run another epoch to get all posteriors
        if np.sum(diffs) < diff_threshold or passes >= max_passes:
            if final_pass:
                # Reset vars before returning
                if auto_reset_shared_vars:
                    for shared_var in shared_variables:
                        shared_var.reset()
                return df_res_final_epoch, df_res_before_convergence, shared_variables
            # if passes % 2 == 1:
            # Convergence must end on a backward pass
            if debug:
                if passes > max_passes:
                    print(
                        f"Alg. didn't converge - Max number of passes reached: {max_passes}, running another epoch to get all posteriors"
                    )
                else:
                    print(
                        f"Alg. converged - All diffs are below {diff_threshold}, running another epoch to get all posteriors"
                    )
            final_pass = True

        # Update variables for the next pass
        passes += 1


def query_back_and_forth_across_days_AR(
    df,
    belief_propagation,
    shared_variables: List[mh.SharedVariableNode],
    variables: List[mh.VariableNode],
    evidence_variables: List[str],
    diff_threshold,
    debug=False,
    auto_reset_shared_vars=True,
    max_passes=99,
    interconnect_AR=True,
):
    """
    algorithm to query the point in time model across days, thus making an approximate longitudinal inference
    the algorithm atlernates forward and backward passes, to ensure that the day-to-day interconnected variables can propagate information backwards as efficiently as possible

    variables: contains the names of the variables to infer, as defined in the graph
    evidence_variables: contains names of the observed variables, as defined in the df's columns
    auto_reset_shared_vars: bool to automatically reset the shared variables after the computations are done
    """
    df = df.sort_values("Date Recorded", ascending=True).reset_index(drop=True)
    # Check that each date in Date Redorded is unique
    assert df["Date Recorded"].nunique() == len(
        df
    ), "Error: Cannot process input df as there are doublons in the Date Recorded column."
    
    final_pass = False
    passes = 1

    df_res_before_convergence = pd.DataFrame({})
    df_res_final_epoch = pd.DataFrame({})

    # Initialize posterior distributions to uniform (for convergence check)
    posteriors_old = [
        get_uniform_message(shared_var.card) for shared_var in shared_variables
    ]

    if interconnect_AR:
        AR = variables[0]
        assert AR.name == "Airway resistance (%)", "AR must be the first variable"

    while True:

        for i in range(len(df)):
            backward_pass = passes % 2 == 0
            if backward_pass:
                """
                NOTE: the last entry will be passed twice, as the last element of the previous pass
                and the first element of the current pass; similar for the first entry. 
                This should not be a problem if the whole algorithm is well implemented, because
                1/ the algorithm is stateful only for the current pass,
                2/ old states get substituted by new states at each pass (including in the aggregated
                virtual messages - which is the tricky part !! - see how SharedVariableNode.set_agg_virtual_message
                and  SharedVariableNode.add_or_update_message are implemented)
                """
                i = len(df) - 1 - i
            row = df.iloc[i]
            date = row["Date Recorded"]
            date_str = date.strftime("%Y-%m-%d")

            # Get query inputs
            evidence_dict = build_evidence(df, i, evidence_variables)
            vevidence_shared = build_virtual_evidence_shared_vars(
                shared_variables, date_str
            )
            vevidence = vevidence_shared

            if interconnect_AR:
                # There is no prev day if it's the first day
                prev_day = None if i - 1 < 0 else df.loc[i - 1, "Date Recorded"]
                # During the first pass, the next day posterior is not available
                next_day = (
                    None
                    if (i + 1 >= len(df) or passes == 1)
                    else df.loc[i + 1, "Date Recorded"]
                )
                vevidence_ar = build_vevidence_ar(AR, date, prev_day, next_day)
                vevidence = vevidence_shared + [vevidence_ar]

            vars_to_infer = get_var_name_list(shared_variables + variables)

            if final_pass:
                # Query all variables to get all posteriors
                if debug:
                    vevidence_str = [(cpd.variable, cpd.values) for cpd in vevidence]
                    print(
                        f"Querying all variables: {vars_to_infer} with evidence: {evidence_dict} and virtual evidence: {vevidence_str}"
                    )
                query_res = belief_propagation.query(
                    vars_to_infer, evidence_dict, vevidence
                )
                df_res_final_epoch = save_res_to_df(
                    df_res_final_epoch,
                    date_str,
                    query_res,
                    vars_to_infer,
                    row,
                    evidence_variables,
                )

            else:
                # Query shared variables to get cross plate message
                if debug:
                    vevidence_str = [(cpd.variable, cpd.values) for cpd in vevidence]
                    print(
                        f"Querying all variables: {vars_to_infer} with evidence: {evidence_dict} and virtual evidence: {vevidence_str}"
                    )
                query_res, query_messages = belief_propagation.query(
                    vars_to_infer,
                    evidence_dict,
                    vevidence,
                    get_messages=True,
                )
                df_res_before_convergence = save_res_to_df(
                    df_res_before_convergence,
                    f"{passes}, {date_str}",
                    query_res,
                    vars_to_infer,
                    row,
                    evidence_variables,
                )

                # Get newly computed message from the query output
                for shared_var in shared_variables:
                    new_message = query_messages[shared_var.factor_node_key]
                    shared_var.add_or_update_message(date_str, new_message)
                    if len(vevidence_shared) == 0:
                        vmessage = get_uniform_message(shared_var.card)
                    else:
                        vmessage = [
                            cpd.values
                            for cpd in vevidence_shared
                            if cpd.variable == shared_var.name
                        ][0]
                    shared_var.set_agg_virtual_message(vmessage, new_message)

            if interconnect_AR:
                AR.add_or_update_posterior(date_str, query_res[AR.name].values)

        posteriors_old, diffs = get_diffs(query_res, posteriors_old, shared_variables)

        for shared_var, diff in zip(shared_variables, diffs):
            print(f"Pass {passes} - Posteriors' diff for {shared_var.name}: {diff}")

        # Convergence reached when the diff is below the threshold
        # or when the maximum number of passes is reached
        # When convergence is reached, run another epoch to get all posteriors
        if np.sum(diffs) < diff_threshold or passes >= max_passes:
            if final_pass:
                # Reset vars before returning
                if auto_reset_shared_vars:
                    for shared_var in shared_variables:
                        shared_var.reset()
                    AR.reset()
                return df_res_final_epoch, df_res_before_convergence, shared_variables
            # if passes % 2 == 1:
            # Convergence must end on a backward pass
            if debug:
                if passes > max_passes:
                    print(
                        f"Alg. didn't converge - Max number of passes reached: {max_passes}, running another epoch to get all posteriors"
                    )
                else:
                    print(
                        f"Alg. converged - All diffs are below {diff_threshold}, running another epoch to get all posteriors"
                    )
            final_pass = True

        # Update variables for the next pass
        passes += 1


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
    Var1,
    Var2,
    df_breathe,
    title,
    colorscale,
    save=False,
):
    layout = [
        [{"type": "scatter", "rowspan": 1}],
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
        title="ecFEV1 (L)",
    )
    plot_scatter(
        fig,
        df_breathe["Date Recorded"],
        df_breathe["ecFEF2575%ecFEV1"],
        row=2,
        col=1,
        colour="black",
        title="ecFEF2575%ecFEV1",
    )
    plot_scatter(
        fig,
        df_breathe["Date Recorded"],
        df_breathe["O2 Saturation"],
        row=3,
        col=1,
        colour="black",
        title="O2 saturation (%)",
    )

    df_res_var1 = get_heatmap_data(df_res_before_convergence, Var1)
    df_res_var2 = get_heatmap_data(df_res_before_convergence, Var2)
    plot_heatmap(fig, df_res_var1, Var1, row=4, col=1, coloraxis="coloraxis1")
    plot_heatmap(fig, df_res_var2, Var2, row=6, col=1, coloraxis="coloraxis2")

    fig.update_layout(
        title=title,
        width=1200,
        height=920,
        font=dict(size=5),
        showlegend=False,
        coloraxis1=dict(
            colorscale=colorscale,
            colorbar_x=1,
            colorbar_y=0.43,
            colorbar_thickness=23,
            colorbar_len=0.305,
        ),
        coloraxis2=dict(
            colorscale=colorscale,
            colorbar_x=1,
            colorbar_y=0.14,
            colorbar_thickness=23,
            colorbar_len=0.305,
        ),
    )
    if save:
        fig.write_image(f"{plotsdir}Validation/{title}.pdf")
    else:
        fig.show()


def plot_query_res(
    df_breathe: pd.DataFrame,
    ecFEV1: mh.VariableNode,
    O2Sat: mh.VariableNode,
    df_query_res: pd.DataFrame,
    AR: mh.VariableNode,
    IA: mh.VariableNode,
    HFEV1: mh.SharedVariableNode,
    HO2Sat: mh.SharedVariableNode,
    title: str,
    colorscale=[[0, "lightcyan"], [0.5, "yellow"], [1, "blue"]],
    save=False,
):
    layout = [
        [{"type": "scatter", "rowspan": 1}, {"type": "scatter", "rowspan": 1}],
        [{"type": "scatter", "rowspan": 1}, {"type": "scatter", "rowspan": 1}],
        [{"type": "heatmap", "rowspan": 2}, {"type": "heatmap", "rowspan": 2}],
        [None, None],
        [{"type": "scatter", "rowspan": 1}, {"type": "scatter", "rowspan": 1}],
        [{"type": "scatter", "rowspan": 1}, None],
    ]
    fig = make_subplots(
        rows=np.shape(layout)[0],
        cols=np.shape(layout)[1],
        specs=layout,
        vertical_spacing=0.05,
    )
    # Priors
    ih.plot_histogram(
        fig, HFEV1, HFEV1.cpt, HFEV1.a, HFEV1.b, 1, 1, HFEV1.name, "#636EFA"
    )
    ih.plot_histogram(
        fig, HO2Sat, HO2Sat.cpt, HO2Sat.a, HO2Sat.b, 1, 2, HO2Sat.name, "#636EFA"
    )

    # Posteriors for shared variables
    hfev1_posterior = df_query_res[HFEV1.name].iloc[-1]
    ih.plot_histogram(
        fig,
        HFEV1,
        hfev1_posterior,
        HFEV1.a,
        HFEV1.b,
        2,
        1,
        HFEV1.name,
        colour="#636EFA",
    )
    ho2sat_posterior = df_query_res[HO2Sat.name].iloc[-1]
    ih.plot_histogram(
        fig,
        HO2Sat,
        ho2sat_posterior,
        HO2Sat.a,
        HO2Sat.b,
        2,
        2,
        HO2Sat.name,
        "#636EFA",
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
        df_breathe["ecFEF2575%ecFEV1"],
        row=6,
        col=1,
        colour="black",
        title="ecFEF2575%ecFEV1",
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

    fig.update_layout(
        title=title,
        height=1050,
        width=1300,
        font=dict(size=5),
        showlegend=False,
        coloraxis1=dict(
            colorscale=colorscale,
            colorbar_x=0.45,
            colorbar_y=0.505,
            colorbar_thickness=23,
            colorbar_len=0.34,
            colorbar={"title": "AR"},
        ),
        coloraxis2=dict(
            colorscale=colorscale,
            colorbar_x=1,
            colorbar_y=0.505,
            colorbar_thickness=23,
            colorbar_len=0.34,
            colorbar={"title": "IA"},
        ),
    )
    if save:
        fig.write_image(f"{plotsdir}Results/{title}.pdf")
    else:
        fig.show()
