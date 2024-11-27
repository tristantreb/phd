import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from scipy.stats import norm

import src.data.helpers as dh
import src.models.helpers as mh


def plot_FEF2575_ratio_with_IA(df, AR_col, FEF2575_col, marginals=False):
    # fig = px.scatter(df, x="AR mean", y="FEF2575%PEF", color="IA mean")
    # fig = px.scatter(df, x="AR mean", y="ecFEF2575%ecPEF", color="IA mean")
    # fig = px.scatter(df, x="AR mean", y="ecFEF2575 % Predicted", color="IA mean")
    # fig = px.scatter(df, x="AR sample", y="ecFEF2575%ecFEV1", color="IA mean")
    # fig = px.scatter(df, x="AR mean", y="FEF2575%FEV1", color="IA mean")
    # Update the scale of the figure's color heatmap
    if marginals:
        fig = px.scatter(
            df,
            x=AR_col,
            y=FEF2575_col,
            color="IA mean",
            marginal_x="histogram",
            marginal_y="histogram",
        )
    else:
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
    fig.update_traces(marker=dict(size=3), selector=dict(mode="markers"))
    fig.update_layout(width=1200, height=800)
    fig.show()


def calc_plot_cpt_ecFEF2575prctecFEV1_given_AR(
    df_sampled,
    df_f3,
    n_samples,
    AR,
    ar_col,
    yVar,
    y_col,
    save=False,
    debug=False,
):
    AR_midbins = np.sort(df_sampled["AR midbin"].unique())

    fig = make_subplots(rows=1, cols=AR.card, shared_yaxes=True)

    cpt_y_var_AR = np.zeros([yVar.card, AR.card])
    cpt_y_var_AR[:] = np.nan

    for idx, midbin in enumerate(AR_midbins):
        values = df_sampled[df_sampled["AR midbin"] == midbin][y_col]
        fig.add_trace(
            go.Histogram(
                y=values,
                ybins=dict(start=yVar.a, end=yVar.b, size=yVar.bin_width),
                histnorm="probability",
            ),
            row=1,
            col=idx + 1,
        )

        if debug:
            print(df_f3.iloc[idx]["mean"])

        # Add gaussian approximation
        p_arr = norm.pdf(
            yVar.midbins,
            loc=df_f3.iloc[idx]["mean"],
            scale=df_f3.iloc[idx]["std"],
        )
        p_arr_norm = p_arr / sum(p_arr)

        cpt_y_var_AR[:, idx] = p_arr_norm

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
                y=yVar.midbins,
                x=cpt_y_var_AR[:, idx_to_plot],
                mode="lines",
                line=dict(color="red"),
            ),
            row=1,
            col=idx + 1,
        )

        fig.update_xaxes(title=f"{midbin}%<br>({values.shape[0]})", row=1, col=idx + 1)

    # Add all remaining CPT entries from idx + 1 to last value
    for j in range(idx + 1, AR.card):
        if debug:
            print(j, idx)
        cpt_y_var_AR[:, j] = cpt_y_var_AR[:, idx - 3]
        fig.add_trace(
            go.Scatter(
                y=yVar.midbins,
                x=cpt_y_var_AR[:, j],
                mode="lines",
                line=dict(color="red"),
            ),
            row=1,
            col=j + 1,
        )
        fig.update_xaxes(title=f"{AR.midbins[j]}%", row=1, col=j + 1, range=[0, 0.4])

    fig.update_yaxes(title=yVar.name, row=1, col=1)
    fig.update_layout(
        width=2000,
        height=400,
        font=dict(size=6),
        showlegend=False,
        title=f"P({yVar.name} | {ar_col})",
    )
    if save:
        fig.write_image(
            f"{dh.get_path_to_main()}PlotsBreathe/AR_modelling/F3 CPT - {y_col} given {ar_col} - {n_samples} samples, {AR.bin_width} bin width.pdf"
        )
    else:
        fig.show()

    return cpt_y_var_AR


def add_traces_F3_mean_and_percentiles_per_AR_bin(fig, df_f3):

    # fig.add_traces(
    #     go.Scatter(
    #         x=df_f3["AR midbin"],
    #         y=df_f3["mean"],
    #         mode="lines+markers",
    #         line=dict(color="blue"),
    #         name="Mean",
    #         yaxis="y2",
    #     )
    # )
    fig.add_traces(
        go.Scatter(
            x=df_f3["AR midbin"],
            y=df_f3["median"],
            mode="lines+markers",
            line=dict(color="blue"),
            name="Median",
            yaxis="y2",
        )
    )
    fig.add_traces(
        go.Scatter(
            x=df_f3["AR midbin"],
            y=df_f3["p16"],
            mode="lines+markers",
            line=dict(color="green"),
            name="16th and 84th percentiles (1 sigma)",
            yaxis="y2",
        )
    )
    fig.add_traces(
        go.Scatter(
            x=df_f3["AR midbin"],
            y=df_f3["p84"],
            mode="lines+markers",
            line=dict(color="green"),
            name="84th percentile",
            legendgroup="1 sigma",
            showlegend=False,
            yaxis="y2",
        )
    )
    fig.add_traces(
        go.Scatter(
            x=df_f3["AR midbin"],
            y=df_f3["p3"],
            mode="lines+markers",
            line=dict(color="red"),
            name="3th and 97th percentiles (2 sigma)",
            yaxis="y2",
        )
    )
    fig.add_traces(
        go.Scatter(
            x=df_f3["AR midbin"],
            y=df_f3["p97"],
            mode="lines+markers",
            line=dict(color="red"),
            name="97th percentile",
            showlegend=False,
            yaxis="y2",
        )
    )
    return -1


def plot_F3_mean_and_percentiles_per_AR_bin(df_f3, ar_col, y_col, save=False):
    fig = go.Figure()

    add_traces_F3_mean_and_percentiles_per_AR_bin(fig, df_f3)

    # Add histogram of the number of contributing data points (df_f3.count) per AR bin
    # Add second y axis for this count histogram
    fig.update_layout()
    fig.add_traces(
        go.Bar(
            x=df_f3["AR midbin"],
            y=df_f3["count"],
            marker=dict(color="lightblue"),
            name="Count",
        )
    )
    fig.data[-1].showlegend = False

    # Add ticks on x axis
    fig.update_xaxes(
        title=f"{ar_col}",
        # tickvals=np.floor(list(df_f3["AR midbin"].values)),
        tickvals=np.arange(1, 91, 2),
        # title_standoff=5,
    )
    fig.update_yaxes(title=y_col)
    title = f"F3 - {y_col} statistics per {ar_col} bin (n={df_f3['count'].sum()})"
    fig.update_layout(
        title=title,
        width=1100,
        height=500,
        yaxis=dict(
            title="Count",
            side="right",
            showgrid=False,
            showline=True,
            showticklabels=True,
            tickmode="sync",
        ),
        yaxis2=dict(
            title=y_col,
            overlaying="y",
            side="left",
            showgrid=True,
            showline=True,
            showticklabels=True,
        ),
        legend=dict(orientation="h", y=1.1),
    )

    if save:
        fig.write_image(f"{dh.get_path_to_main()}PlotsBreathe/AR_modelling/{title}.pdf")
    else:
        fig.show()
    return -1


def get_sampled_df_and_statistics_df(
    df, n_samples, AR, y_col="ecFEF2575%ecFEV1", AR_bin_width=None
):
    if AR_bin_width is None:
        AR_bin_width = AR.bin_width
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
            np.ceil(max(df_sampled["AR sample"])) + AR_bin_width,
            AR_bin_width,
        ),
    )

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
            count=(y_col, "count"),
        )
        .reset_index()
    )

    df_sampled["AR midbin"] = df_sampled["AR bin"].apply(
        lambda x: x.left + AR_bin_width / 2
    )
    df_f3["AR midbin"] = df_f3["AR bin"].apply(lambda x: x.left + AR_bin_width / 2)
    return df_sampled, df_f3


def get_sampled_df_for_AR_IA(df, n_samples, AR, IA):
    df_sampled = df.copy()

    # Renormalise all AR distributions
    df_sampled["AR norm"] = df_sampled.apply(
        lambda row: row.AR / sum(row["AR"]), axis=1
    )

    # Create n AR samples per row
    df_sampled["AR sample"] = df_sampled.apply(
        lambda row: AR.sample(n=n_samples, p=row["AR norm"]), axis=1
    )
    df_sampled["IA sample"] = df_sampled.apply(
        lambda row: IA.sample(n=n_samples, p=row["IA"]), axis=1
    )

    df_sampled = df_sampled.explode(["AR sample", "IA sample"]).reset_index(drop=True)

    print(f'Max sampled AR values: {max(df_sampled["AR sample"]):.2f}')
    print(f'Max sampled IA values: {max(df_sampled["IA sample"]):.2f}')

    return df_sampled


def add_binned_up_var(df_sampled, col, var_name, bin_width):
    df_sampled[f"{var_name} bin"] = pd.cut(
        df_sampled[col],
        bins=np.arange(
            np.floor(min(df_sampled[col])),
            np.ceil(max(df_sampled[col])) + bin_width,
            bin_width,
        ),
    )

    df_sampled[f"{var_name} midbin"] = df_sampled[f"{var_name} bin"].apply(
        lambda x: x.left + bin_width / 2
    )
    return df_sampled


def get_sampled_df_and_statistics_df_for_IA(df, n_samples, AR, AR_bin_width, IA):
    df_sampled = df.copy()

    df_sampled = get_sampled_df_for_AR_IA(df, n_samples, AR, IA)

    df_sampled = add_binned_up_var(df_sampled, "AR sample", "AR", AR_bin_width)
    return df_sampled


def model_f3(df, AR, ar_col, y_col="ecFEF2575%ecFEV1", save=True):
    df["ecFEF2575%ecFEVmodel_f31"] = df["ecFEF2575"] / df["ecFEV1"] * 100

    # Parameters
    n_samples = 100

    ecFEF2575prctecFEV1 = mh.VariableNode(
        "ecFEF25-75 % ecFEV1 (%)", 0, 200, 2, prior=None
    )

    df_sampled, df_f3 = get_sampled_df_and_statistics_df(df, n_samples, AR)

    plot_F3_mean_and_percentiles_per_AR_bin(df_f3, ar_col, y_col, save=save)
    cpt_f3 = calc_plot_cpt_ecFEF2575prctecFEV1_given_AR(
        df_sampled, df_f3, n_samples, AR, ar_col, ecFEF2575prctecFEV1, y_col, save=save
    )
    return cpt_f3, df_f3, df_sampled


def fit_ia_hist_profile(x, y, IA, debug=False):
    # def func(x, A, K):
    #     return A * np.exp(K * x) + 0.001

    # def objective(params, x, y):
    #     return np.sum((func(x, *params) - y) ** 2)

    # # Initial guess for parameters
    # initial_guess = [1, -1]

    # # Minimize the objective function with the constraint
    # result = minimize(objective, initial_guess, args=(x, y), constraints=())
    # A, K = result.x

    # if debug:
    #     print(f"A: {A:.5f}, K: {K:.5f}")x

    # y_fit = func(IA.midbins, A, K)

    def func(x, A, K, C):
        return A * np.exp(-x / K) + C

    def objective(params, x, y):
        return np.sum((func(x, *params) - y) ** 2)

    # Initial guess for parameters
    initial_guess = [1, 1, 0]

    # Minimize the objective function with the constraint
    result = minimize(objective, initial_guess, args=(x, y), constraints=())
    A, K, C = result.x

    if debug:
        print(f"A: {A:.5f}, K: {K:.5f}, C: {C:.5f}")

    y_fit = func(IA.midbins, A, K, C)
    return y_fit


def calc_plot_cpt_IA_given_AR(
    df_sampled, AR, AR_bin_width, ar_col, IA, n_samples, save=False, debug=False
):
    cpt_ia_ar = np.zeros([IA.card, AR.card])

    ar_groups = np.sort(list(df_sampled["AR midbin"].unique()))

    # Bin sampled IA values
    IA_bin_width = 2
    df_sampled["IA bin"] = pd.cut(
        df_sampled["IA sample"],
        bins=np.arange(
            np.floor(min(df_sampled["IA sample"])),
            np.ceil(max(df_sampled["IA sample"])) + IA_bin_width,
            IA_bin_width,
        ),
    )
    df_sampled["IA midbin"] = df_sampled["IA bin"].apply(
        lambda x: x.left + IA_bin_width / 2
    )

    fig = make_subplots(rows=1, cols=len(ar_groups), shared_yaxes=True)

    for idx, ar_group in enumerate(ar_groups):
        df_tmp = df_sampled[df_sampled["AR midbin"] == ar_group].copy()

        # Create histogram data for IA, binned by IA bins
        s_ia_hist = df_tmp["IA midbin"].value_counts()
        # Add 10% of the data distributed evenly on each bin
        print("dirichlet factor", max(100, round(s_ia_hist.sum() * 0.1 / IA.card)))
        s_ia_hist_dirichlet = s_ia_hist + round(s_ia_hist.sum() * 0.1 / IA.card)
        s_ia_hist_norm = s_ia_hist_dirichlet / s_ia_hist_dirichlet.sum()

        x = np.array(s_ia_hist_norm.index)
        y = s_ia_hist_norm.values

        y_fit = fit_ia_hist_profile(x, y, IA, debug)

        # If the bin_width used for the fit it greater than the variable's bin_width, use the same distribution for all variables bins contributing to the fit-bins
        for i in range(idx * IA_bin_width, (idx + 1) * IA_bin_width):
            cpt_ia_ar[:, i] = y_fit

        # Add trace with fitted exponential
        fig.add_trace(
            go.Scatter(
                y=IA.midbins,
                x=y_fit,
                mode="lines",
                name="Fitted exponential",
                marker=dict(color="red"),
            ),
            row=1,
            col=idx + 1,
        )

        fig.add_trace(
            go.Scatter(
                y=x,
                x=y,
                mode="markers",
                name="Real values",
                marker=dict(color="blue"),
            ),
            row=1,
            col=idx + 1,
        )

        fig.update_xaxes(
            range=[-0.02, 1],
            title=f"{ar_group}%<br>(n={df_tmp['IA sample'].shape[0]})",
            row=1,
            col=idx + 1,
        )
    fig.update_traces(marker=dict(size=4), line=dict(width=2))
    fig.update_yaxes(title=IA.name, row=1, col=1, tickvals=np.linspace(IA.a, IA.b, 16))
    fig.update_yaxes(tickvals=np.linspace(IA.a, IA.b, 16))
    fig.update_layout(
        width=2000,
        height=400,
        font=dict(size=8),
        showlegend=False,
        title=f"P({IA.name} | {ar_col})",
    )
    # Add overacrhiing x axes title
    fig.add_annotation(
        x=0.5,
        y=-0.36,
        xref="paper",
        yref="paper",
        text=f"{ar_col}%",
        showarrow=False,
        font=dict(size=12),
    )
    if save:
        fig.write_image(
            f"{dh.get_path_to_main()}PlotsBreathe/AR_modelling/CPT - IA given {ar_col} - {n_samples} samples, {AR_bin_width} AR bin width.pdf"
        )
    else:
        fig.show()
    return cpt_ia_ar


def check_IA_cpt(cpt, IA, AR, debug=False):
    # Check cpt is correct 1: 2 consecutive values are equal and after 66, all are the same
    idx_sixty_six_from_back = AR.card - AR.get_bin_for_value(66)[1]
    if debug:
        print("CPT is correct if 2 consecutive values are equal")
        for idx in range(AR.card - 1):
            print(
                "bin",
                AR.get_bins_str()[idx],
                "- bin",
                AR.get_bins_str()[idx + 1],
                "cpt val",
                cpt[0, idx] - cpt[0, idx + 1],
            )
        print("CPT is correct if after 66 % of AR all are the same")

        for idx in range(idx_sixty_six_from_back - 1):
            print(
                "bin",
                AR.get_bins_str()[-idx - 1],
                "- bin",
                AR.get_bins_str()[-idx_sixty_six_from_back],
                "cpt val",
                cpt[0, -idx - 1] - cpt[0, -idx_sixty_six_from_back],
            )
    else:
        print("CPT is correct if 2 consecutive values are equal")
        for idx in np.arange(0, AR.card - 1, 2):
            assert (
                cpt[0, idx] - cpt[0, idx + 1] == 0
            ), f"Error at bin {idx}: {cpt[0, idx]} diff from {cpt[0, idx + 1]}"
        print("CPT is correct if after 66 % of AR all are the same")

        for idx in range(12):
            assert (
                cpt[0, -idx - 1] - cpt[0, -idx_sixty_six_from_back] == 0
            ), f"Error at bin {-idx-1}: {cpt[0, -idx-11]} diff from {cpt[0, -idx_sixty_six_from_back]}"

    # Check that values are decreasing
    print("Check that IA values are decreasing with increasing AR bin")
    if debug:
        for idx in range(IA.card - 1):
            print("idx", idx, "cpt val", cpt[idx, 0] - cpt[idx + 1, 0])
    else:
        for idx in range(IA.card - 1):
            assert (
                cpt[idx, 0] - cpt[idx + 1, 0] > 0
            ), f"Error at bin {idx}: {cpt[idx, 0]} diff from {cpt[idx + 1, 0]}"
