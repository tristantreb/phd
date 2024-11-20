import concurrent.futures
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import src.data.breathe_data as bd
import src.data.helpers as dh
import src.inference.helpers as ih
import src.models.builders as mb

df = bd.load_meas_from_excel("BR_O2_FEV1_FEF2575_conservative_smoothing_with_idx")


def compute_log_p_D_given_M_per_entry_per_HFEV1_obs(
    df_for_ID_in, debug=False, save=False, ar_prior="uniform"
):
    df_for_ID_in = df_for_ID_in.copy().reset_index(drop=True)
    id = df_for_ID_in.loc[0, "ID"]
    height = df_for_ID_in.loc[0, "Height"]
    age = df_for_ID_in.loc[0, "Age"]
    sex = df_for_ID_in.loc[0, "Sex"]
    print(f"Processing ID {id}")

    (
        _,
        inf_alg,
        HFEV1,
        uecFEV1,
        ecFEV1,
        AR,
        _,
        _,
        _,
        _,
        _,
        ecFEF2575prctecFEV1,
    ) = mb.o2sat_fev1_fef2575_point_in_time_model_noise_shared_healthy_vars_light(
        height, age, sex, ar_prior=ar_prior
    )

    # HFEV1 can't be above max observed ecFEV1
    HFEV1_obs_list = HFEV1.midbins
    print(f"ID {id}")

    N = len(df_for_ID_in)
    df_for_ID = df_for_ID_in.copy()

    H = len(HFEV1_obs_list)
    log_p_D_given_M = np.zeros((N, H))
    AR_dist_given_M_matrix = np.zeros((N, AR.card, H))

    # Get the joint probability of ecFEV1 and ecFEF2575 given the model for this individual
    # For each entry
    for n, row in df_for_ID.iterrows():
        if debug:
            print(f"Processing row {n+1}/{N}")

        # For each model given an HFEV1 observation
        for h, HFEV1_obs in enumerate(HFEV1_obs_list):

            # Getting the joint probabilities of ecFEF2575 and ecFEV1 under the model
            res1, _ = ih.infer_on_factor_graph(
                inf_alg,
                [ecFEV1],
                [[HFEV1, HFEV1_obs]],
                get_messages=True,
            )
            dist_ecFEV1 = res1[ecFEV1.name].values

            # Observe both HFEV1 and ecFEV1 to compute the joint probability
            # P(ecFEV1, ecFEF2575 | HFEV1) = P(ecFEV1 | HFEV1) * P( ecFEF2575 | HFEV1, ecFEV1)
            res2, _ = ih.infer_on_factor_graph(
                inf_alg,
                [ecFEF2575prctecFEV1],
                [[HFEV1, HFEV1_obs], [ecFEV1, row.ecFEV1]],
                get_messages=True,
            )
            dist_ecFEF2575prctecFEV1 = res2[ecFEF2575prctecFEV1.name].values

            res3 = ih.infer_on_factor_graph(
                inf_alg,
                [AR],
                [
                    [HFEV1, HFEV1_obs],
                    [ecFEV1, row.ecFEV1],
                    [ecFEF2575prctecFEV1, row["ecFEF2575%ecFEV1"]],
                ],
                get_messages=False,
            )
            dist_AR = res3[AR.name].values

            # The probability of the data given the model is the expectation of the data given the model
            idx_obs_ecFEV1 = ecFEV1.get_bin_for_value(row.ecFEV1)[1]
            idx_obs_ecFEF2575 = ecFEF2575prctecFEV1.get_bin_for_value(
                row["ecFEF2575%ecFEV1"]
            )[1]

            # Get the probability of the data given the model
            p_ecFEV1 = dist_ecFEV1[idx_obs_ecFEV1]
            p_ecFEF2575 = dist_ecFEF2575prctecFEV1[idx_obs_ecFEF2575]

            # Save information for this round
            AR_dist_given_M_matrix[n, :, h] = dist_AR
            log_p_D_given_M[n, h] = np.log(p_ecFEV1) + np.log(p_ecFEF2575)

    if debug:
        print("log(P(D|M)), first row", log_p_D_given_M[0, :])

    # For each HFEV1 model, given HFEV1_obs_list, we compute the log probability of the model given the data
    # log(P(M|D)) = 1/N * sum_n log(P(D|M)) + Cn_avg + log(P(M))
    log_p_M_given_D = np.zeros(H)
    for h, HFEV1_obs in enumerate(HFEV1_obs_list):
        log_p_M = np.log(HFEV1.cpt[HFEV1.get_bin_for_value(HFEV1_obs)[1]])
        log_p_M_given_D[h] = np.sum(log_p_D_given_M[:, h]) + log_p_M

    # Exponentiating very negative numbers gives too small numbers
    # Setting the highest number to 1
    shift = 1 - log_p_M_given_D.max()
    log_p_M_given_D_shifted = log_p_M_given_D + shift

    # Exponentiate and normalise
    p_M_given_D = np.exp(log_p_M_given_D_shifted)
    p_M_given_D = p_M_given_D / p_M_given_D.sum()

    # Add plot
    layout = [
        [{"type": "scatter", "rowspan": 1, "colspan": 1}, None, None],
        [{"type": "heatmap", "rowspan": 3, "colspan": 3}, None, None],
        [None, None, None],
        [None, None, None],
    ]
    fig = make_subplots(
        rows=np.shape(layout)[0],
        cols=np.shape(layout)[1],
        specs=layout,
        vertical_spacing=0.1,
    )

    # Add HFEV1 posterior
    ih.plot_histogram(fig, HFEV1, p_M_given_D, 0, 6, 1, 1, annot=True)

    # Add heatmap with AR posteriors
    AR_dist_given_M_matrix = np.matmul(AR_dist_given_M_matrix, p_M_given_D)
    df1 = pd.DataFrame(
        data=AR_dist_given_M_matrix,
        columns=AR.get_bins_str(),
        index=df_for_ID_in["Date Recorded"].apply(
            lambda date: date.strftime("%Y-%m-%d")
        ),
    )
    colorscale = [
        [0, "white"],
        [0.01, "red"],
        [0.05, "yellow"],
        [0.1, "cyan"],
        [0.6, "blue"],
        [1, "black"],
    ]

    fig.add_trace(
        go.Heatmap(z=df1.T, x=df1.index, y=df1.columns, coloraxis="coloraxis1"),
        row=2,
        col=1,
    )

    title = f"{id} - Posterior HFEV1 after fusing all P(M_h|D)<br>AR prior: {ar_prior}"
    fig.update_layout(
        font=dict(size=12),
        height=700,
        width=1200,
        title=title,
        coloraxis1=dict(
            colorscale=colorscale,
            colorbar_x=1,
            colorbar_y=0.36,
            # colorbar_thickness=23,
            colorbar_len=0.77,
        ),
    )
    # Add Date on x axis
    fig.update_xaxes(title_text=HFEV1.name, row=1, col=1)
    fig.update_yaxes(title_text="p", row=1, col=1)
    fig.update_yaxes(title_text=AR.name, row=2, col=1)
    fig.update_xaxes(
        title_text="Date",
        row=2,
        col=1,
        nticks=50,
        type="category",
    )

    if save:
        fig.write_image(
            dh.get_path_to_main() + f"/PlotsBreathe/Cutset_conditioning/{title}.png",
            scale=3,
        )
    else:
        fig.show()

    return fig, p_M_given_D, AR_dist_given_M_matrix


def process_id(id):
    ar_prior = "breathe (2 days model, ecFEV1, ecFEF25-75)"
    ar_prior = "uniform"

    dftmp = df[df.ID == id]
    return compute_log_p_D_given_M_per_entry_per_HFEV1_obs(
        dftmp, debug=False, save=True, ar_prior=ar_prior
    )


# Run the function in parallel using ProcessPoolExecutor
if __name__ == "__main__":

    interesting_ids = [
        "132",
        "146",
        "177",
        "180",
        "202",
        "527",
        "117",
        "131",
        "134",
        "191",
        "139",
        "253",
        "101",
        # Also from consec values
        "405",
        "272",
        "201",
        "203",
    ]

    # num_cores = os.cpu_count()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Map the function to the list of unique IDs
        list(executor.map(process_id, interesting_ids))
