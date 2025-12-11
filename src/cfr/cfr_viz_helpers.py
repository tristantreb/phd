import numpy as np
import plotly.graph_objects as go

import src.models.helpers as mh


def get_dumbell_plot_data(df, ac_row, AC):
    # Avoid modifying the original dataframe
    df_res = df.copy()

    # Assuming AC methods can take the row value directly
    mean = df_res[ac_row].apply(AC.get_mean)
    std = df_res[ac_row].apply(AC.get_std)

    df_res[f"{ac_row} mean"] = mean
    df_res[f"{ac_row} low"] = mean - std
    df_res[f"{ac_row} high"] = mean + std

    # Get Sorted IDs
    ids_sorted = df_res.sort_values("ecFEV1 % Predicted", ascending=False)["ID"].values
    # Sort by diff
    df_res['Healhtier diff'] = df_res[f"{AC.name} mean"] - df_res['ecFEV1 % Predicted']
    ids_sorted = df_res.sort_values('Healhtier diff')["ID"].values

    # Prepare for Plotting
    # Map 'low' and 'high' to the same name ('dist') to group them using melt
    plot_cols = {
        f"{ac_row} low": f"{ac_row} dist",
        f"{ac_row} high": f"{ac_row} dist",
    }

    df_melted = (
        df_res.rename(columns=plot_cols)
        .melt(
            id_vars=["ID"],
            value_vars=["ecFEV1 % Predicted", f"{ac_row} dist", f"{ac_row} mean"],
            var_name="measure",
            value_name="value",
        )
        .set_index("ID")
        .loc[ids_sorted]
        .reset_index()
    )

    return df_melted, df_res, ids_sorted


def plot_dumbell_for_df(fig, df, measures, col):
    ac_dist = measures[0] # sigma up, sigma down
    ac_mean = measures[1]
    baseline = measures[2]

    mask = df["measure"] == ac_dist
    for id in df[mask]["ID"].unique():
        mask_id = df["ID"] == id
        # Add mask
        mask_final = mask_id & mask
        fig.add_trace(
            go.Scatter(
                x=df[mask_final]["value"],
                y=df[mask_final]["ID"],
                mode="lines",
                name="ecFEV1 % healthy FEV1 * f(FEF25-75)",
                marker=dict(color="red"),
                line=dict(width=3),
                showlegend=(True if id == df["ID"].unique()[0] else False),
            ),
            row=1,
            col=col,
        )


    mask = df["measure"] == ac_mean
    fig.add_trace(
        go.Scatter(
            x=df[mask]["value"],
            y=df[mask]["ID"],
            mode="markers",
            marker=dict(color="black", size=2),
            showlegend=(True if id == df["ID"].unique()[0] else False),
        ),
        row=1,
        col=col,
    )

    mask = df["measure"] == baseline
    ecfev1_prct_pred = df[mask]["value"]
    # Where above 100, set to 100
    # ecfev1_prct_pred = np.clip(ecfev1_prct_pred, 0, 100)
    fig.add_trace(
        go.Scatter(
            x=ecfev1_prct_pred,
            y=df[mask]["ID"],
            mode="markers",
            name="ecFEV1%Predicted",
            marker=dict(size=4, color="blue"),
        ),
        row=1,
        col=col,
    )
