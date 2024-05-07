import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

import src.data.helpers as dh


def plot_FEF2575_ratio_with_IA(df, AR_col, FEF2575_col):
    # fig = px.scatter(df, x="AR mean", y="FEF2575%PEF", color="IA mean")
    # fig = px.scatter(df, x="AR mean", y="ecFEF2575%ecPEF", color="IA mean")
    # fig = px.scatter(df, x="AR mean", y="ecFEF2575 % Predicted", color="IA mean")
    # fig = px.scatter(df, x="AR sample", y="ecFEF2575%ecFEV1", color="IA mean")
    # fig = px.scatter(df, x="AR mean", y="FEF2575%FEV1", color="IA mean")
    # Update the scale of the figure's color heatmap
    fig = px.scatter(df, x=AR_col, y=FEF2575_col, color="IA mean")
    t1 = 2
    t2 = 8
    colorscale = [
        [0, "#FFFFA1"],
        [round(t1 / max(df["IA mean"]), 2) * 0.99, "#FFFFA1"],
        [round(t1 / max(df["IA mean"]), 2), "orange"],
        [round(t2 / max(df["IA mean"]), 2) * 0.99, "orange"],
        [round(t2 / max(df["IA mean"]), 2), "black"],
        [1, "black"],
    ]
    fig.update_coloraxes(colorscale=colorscale)
    # fig.update_yaxes(range=[0,150])
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(width=1200, height=800)
    fig.show()


def calc_and_plot_FEF2575prctFEV1_AR_cpt(
    df_sampled,
    df_f3,
    n_samples,
    AR,
    ecFEF2575prctecFEV1,
    ar_col,
    save=False,
    debug=False,
):
    AR_midbins = np.sort(df_sampled["AR midbin"].unique())

    fig = make_subplots(rows=1, cols=len(AR.bins), shared_yaxes=True)

    cpt_AR_FEF2575prctFEV1 = np.empty([len(ecFEF2575prctecFEV1.bins), len(AR.bins)])
    cpt_AR_FEF2575prctFEV1[:] = np.nan

    for idx, midbin in enumerate(AR_midbins):
        values = df_sampled[df_sampled["AR midbin"] == midbin]["ecFEF2575%ecFEV1"]
        fig.add_trace(
            go.Histogram(
                y=values,
                ybins=dict(start=0, end=200, size=ecFEF2575prctecFEV1.bin_width),
                histnorm="probability",
            ),
            row=1,
            col=idx + 1,
        )

        if debug:
            print(df_f3.iloc[idx]["mean"])

        # Add gaussian approximation
        p_arr = norm.pdf(
            ecFEF2575prctecFEV1.midbins,
            loc=df_f3.iloc[idx]["mean"],
            scale=df_f3.iloc[idx]["std"],
        )
        p_arr_norm = p_arr / sum(p_arr)

        cpt_AR_FEF2575prctFEV1[:, idx] = p_arr_norm

        # The 3 last bins are unreliable (too few data + mean is increase instead of decreasing)
        # Instead use the 4th last bin, i.e. the last reliable bin
        if idx < len(AR_midbins) - 3:
            idx_to_plot = idx
        else:
            idx_to_plot = len(AR_midbins) - 4

        if debug:
            print(idx, idx_to_plot)
        fig.add_trace(
            go.Scatter(
                y=ecFEF2575prctecFEV1.midbins,
                x=cpt_AR_FEF2575prctFEV1[:, idx_to_plot],
                mode="lines",
                line=dict(color="red"),
            ),
            row=1,
            col=idx + 1,
        )

        fig.update_xaxes(title=f"{midbin}%<br>({values.shape[0]})", row=1, col=idx + 1)

    # Add all remaining CPT entries from idx + 1 to last value
    for j in range(idx + 1, len(AR.bins)):
        if debug:
            print(j, idx)
        cpt_AR_FEF2575prctFEV1[:, j] = cpt_AR_FEF2575prctFEV1[:, idx - 3]
        fig.add_trace(
            go.Scatter(
                y=ecFEF2575prctecFEV1.midbins,
                x=cpt_AR_FEF2575prctFEV1[:, j],
                mode="lines",
                line=dict(color="red"),
            ),
            row=1,
            col=j + 1,
        )
        fig.update_xaxes(title=f"{AR.midbins[j]}%", row=1, col=j + 1, range=[0, 0.4])

    fig.update_yaxes(title=ecFEF2575prctecFEV1.name, row=1, col=1)
    fig.update_layout(
        width=2000,
        height=400,
        font=dict(size=6),
        showlegend=False,
        title=f"P({ecFEF2575prctecFEV1.name} | {ar_col})",
    )
    if save:
        fig.write_image(
            f"{dh.get_path_to_main()}PlotsBreathe/AR_modelling/F3 CPT - ecFEF2575%ecFEV1 given {ar_col} - {n_samples} samples, {AR.bin_width} bin width.pdf"
        )
    else:
        fig.show()

    return cpt_AR_FEF2575prctFEV1


def plot_F3_mean_and_percentiles_per_AR_bin(df_f3, title, save=False):
    fig = go.Figure()
    fig.add_traces(
        go.Scatter(
            x=df_f3["AR midbin"],
            y=df_f3["mean"],
            mode="lines+markers",
            line=dict(color="blue"),
            name="Mean",
        )
    )
    fig.add_traces(
        go.Scatter(
            x=df_f3["AR midbin"],
            y=df_f3["median"],
            mode="lines+markers",
            line=dict(color="purple"),
            name="Median",
        )
    )
    fig.add_traces(
        go.Scatter(
            x=df_f3["AR midbin"],
            y=df_f3["p16"],
            mode="lines+markers",
            line=dict(color="red"),
            name="16th percentile",
        )
    )
    fig.add_traces(
        go.Scatter(
            x=df_f3["AR midbin"],
            y=df_f3["p84"],
            mode="lines+markers",
            line=dict(color="red"),
            name="84th percentile",
        )
    )
    fig.add_traces(
        go.Scatter(
            x=df_f3["AR midbin"],
            y=df_f3["p3"],
            mode="lines+markers",
            line=dict(color="green"),
            name="3th percentile",
        )
    )
    fig.add_traces(
        go.Scatter(
            x=df_f3["AR midbin"],
            y=df_f3["p97"],
            mode="lines+markers",
            line=dict(color="green"),
            name="97th percentile",
        )
    )
    # Add ticks on x axis
    fig.update_xaxes(
        title="Airway resistance midbin (%)",
        tickvals=np.floor(list(df_f3["AR midbin"].values)),
    )
    fig.update_yaxes(title="ecFEF2575%ecFEV1 (%)")
    fig.update_layout(title=title, width=1100, height=500)
    if save:
        fig.write_image(f"{dh.get_path_to_main()}PlotsBreathe/AR_modelling/{title}.pdf")
    else:
        fig.show()
    return -1


def get_sampled_df_and_statistics_df(df, n_samples, AR):
    df_sampled = df.copy()
    df_sampled["AR sampled"] = np.nan

    # Renormalise all AR distributions
    df_sampled["AR norm"] = df_sampled.apply(
        lambda row: row.AR / sum(row["AR"]), axis=1
    )

    # Create n AR samples per row
    df_sampled["AR sample"] = df_sampled.apply(
        lambda row: AR.sample(n=n_samples, p=row["AR norm"]), axis=1
    )

    df_sampled = df_sampled.explode("AR sample").reset_index(drop=True)

    print(f'Max sampled AR values: {max(df_sampled["AR sample"]):.2f}')

    df_sampled["AR bin"] = pd.cut(
        df_sampled["AR sample"],
        bins=np.arange(
            np.floor(min(df_sampled["AR sample"])),
            np.ceil(max(df_sampled["AR sample"])) + AR.bin_width,
            AR.bin_width,
        ),
    )

    y_col = "ecFEF2575%ecFEV1"
    df_f3 = (
        df_sampled.groupby("AR bin")
        .agg(
            mean=(y_col, "mean"),
            std=(y_col, "std"),
            median=(y_col, "median"),
            p3=(y_col, lambda x: np.percentile(x, 3)),
            p97=(y_col, lambda x: np.percentile(x, 97)),
            p16=(y_col, lambda x: np.percentile(x, 16)),
            p84=(y_col, lambda x: np.percentile(x, 84)),
        )
        .reset_index()
    )

    df_sampled["AR midbin"] = df_sampled["AR bin"].apply(
        lambda x: x.left + AR.bin_width / 2
    )
    df_f3["AR midbin"] = df_f3["AR bin"].apply(lambda x: x.left + AR.bin_width / 2)
    return df_sampled, df_f3
