import concurrent.futures
import itertools
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import src.data.breathe_data as bd
import src.data.helpers as dh
import src.inference.helpers as ih
import src.modelling_ar.ar as ar
import src.inf_cutset_conditioning.helpers as cutseth
import src.models.builders as mb

df = bd.load_meas_from_excel("BR_O2_FEV1_FEF2575_conservative_smoothing_with_idx")


def compute_log_p_D_given_M_per_HFEV1_HO2Sat_obs_temporal_ARfinal(
    df_for_ID_in,
    ar_prior,
    ar_change_cpt_suffix,
    debug=False,
    save=False,
):
    df_for_ID_in = (
        df_for_ID_in.copy()
        .sort_values(by="Date Recorded", ascending=True)
        .reset_index(drop=True)
    )
    id = df_for_ID_in.loc[0, "ID"]
    height = df_for_ID_in.loc[0, "Height"]
    age = df_for_ID_in.loc[0, "Age"]
    sex = df_for_ID_in.loc[0, "Sex"]

    (
        _,
        _,
        HFEV1,
        _,
        _,
        HO2Sat,
        _,
        _,
        _,
        _,
        _,
    ) = mb.o2sat_fev1_fef2575_long_model_shared_healthy_vars_and_temporal_ar(
        height, age, sex, ar_change_cpt_suffix=ar_change_cpt_suffix
    )

    # HFEV1 can't be above max observed ecFEV1
    HFEV1_obs_list = HFEV1.midbins[
        HFEV1.midbins - HFEV1.bin_width / 2 >= df_for_ID_in.ecFEV1.max()
    ]
    # Create tuples of obs (HFEV1, HO2Sat) to observe
    H_obs_list = [
        list(zip([HFEV1_obs] * HO2Sat.card, HO2Sat.midbins))
        for HFEV1_obs in HFEV1_obs_list
    ]
    # Flatten the list
    H_obs_list = list(itertools.chain(*H_obs_list))

    (
        _,
        inf_alg,
        HFEV1,
        ecFEV1,
        AR,
        HO2Sat,
        O2SatFFA,
        IA,
        UO2Sat,
        O2Sat,
        ecFEF2575prctecFEV1,
    ) = mb.o2sat_fev1_fef2575_long_model_shared_healthy_vars_and_temporal_ar(
        height,
        age,
        sex,
        ia_prior="uniform",
        ar_prior=ar_prior,
        ar_change_cpt_suffix=ar_change_cpt_suffix,
        n_cutset_conditioned_states=len(H_obs_list),
    )

    print(
        f"ID {id} - Number of HFEV1, HO2Sat specific models: {len(H_obs_list)}, max ecFEV1: {df_for_ID_in.ecFEV1.max()}, first possible bin for HFEV1: {HFEV1.get_bin_for_value(HFEV1_obs_list[0])[0]}"
    )

    N = len(df_for_ID_in)
    df_for_ID = df_for_ID_in.copy()
    H = len(H_obs_list)
    log_p_D_given_M = np.zeros((N, H))
    AR_dist_given_M_matrix = np.zeros((N, AR.card, H))

    # Get the joint probability of ecFEV1 and ecFEF2575 given the model for this individual
    # For each entry
    tic = time.time()
    for n, row in df_for_ID.iterrows():
        if debug:
            print(f"Processing row {n+1}/{N}")

        # There is no prev day if it's the first day
        prev_date = None if n - 1 < 0 else df_for_ID.loc[n - 1, "Date Recorded"]
        # During the first pass, the next day posterior is not available
        # No next date for now because we just go forward once.
        # next_date = None if i + 1 >= len(df_for_ID) else df_for_ID.loc[i + 1, "Date Recorded"]

        # For each model given an HFEV1 observation
        for h, (HFEV1_obs, HO2Sat_obs) in enumerate(H_obs_list):

            vevidence_ar = cutseth.build_vevidence_cutset_conditioned_ar(AR, h, prev_date, None)

            # Getting the joint probabilities of ecFEF2575 and ecFEV1 under the model
            res1 = ih.infer_on_factor_graph(
                inf_alg,
                [ecFEV1],
                [[HFEV1, HFEV1_obs], [HO2Sat, HO2Sat_obs]],
                [vevidence_ar],
            )
            dist_ecFEV1 = res1[ecFEV1.name].values

            # Observe both HFEV1 and ecFEV1 to compute the joint probability
            # P(ecFEV1, ecFEF2575, O2sat | HFEV1) = P(ecFEV1 | HFEV1) * P( ecFEF2575 | HFEV1, ecFEV1) * P( O2Sat | HFEV1, ecFEV1, ecFEF2575)
            res2 = ih.infer_on_factor_graph(
                inf_alg,
                [ecFEF2575prctecFEV1, AR],
                [[HFEV1, HFEV1_obs], [HO2Sat, HO2Sat_obs], [ecFEV1, row.ecFEV1]],
                [vevidence_ar],
            )
            dist_ecFEF2575prctecFEV1 = res2[ecFEF2575prctecFEV1.name].values

            # res3, _ = ih.infer_on_factor_graph(
            #     inf_alg,
            #     [O2Sat],
            #     [
            #         [HFEV1, HFEV1_obs],
            #        [HO2Sat, HO2Sat_obs],
            #         [ecFEV1, row.ecFEV1],
            #         [ecFEF2575prctecFEV1, row["ecFEF2575%ecFEV1"]],
            #     ],
            #     get_messages=True,
            # )
            # dist_O2Sat = res3[O2Sat.name].values

            # res4, _ = ih.infer_on_factor_graph(
            #     inf_alg,
            #     [AR],
            #     [
            #         [HFEV1, HFEV1_obs],
            #         [HO2Sat, HO2Sat_obs],
            #         [ecFEV1, row.ecFEV1],
            #         [ecFEF2575prctecFEV1, row["ecFEF2575%ecFEV1"]],
            #         # [O2Sat, row["O2 Saturation"]],
            #     ],
            #     get_messages=True,
            # )

            # Use previously inferred AR, and add message from FEF25-75
            m_to_factor = ecFEF2575prctecFEV1.get_point_message(row["ecFEF2575%ecFEV1"])
            factor_to_AR = np.matmul(m_to_factor, ecFEF2575prctecFEV1.cpt)
            factor_to_AR = factor_to_AR / factor_to_AR.sum()

            dist_AR = res2[AR.name].values * factor_to_AR
            dist_AR = dist_AR / dist_AR.sum()

            # The probability of the data given the model is the expectation of the data given the model
            idx_obs_ecFEV1 = ecFEV1.get_bin_for_value(row.ecFEV1)[1]
            idx_obs_ecFEF2575 = ecFEF2575prctecFEV1.get_bin_for_value(
                row["ecFEF2575%ecFEV1"]
            )[1]
            # idx_obs_O2Sat = O2Sat.get_bin_for_value(row["O2 Saturation"])[1]

            # Get the probability of the data given the model
            p_ecFEV1 = dist_ecFEV1[idx_obs_ecFEV1]
            p_ecFEF2575 = dist_ecFEF2575prctecFEV1[idx_obs_ecFEF2575]
            # p_O2Sat = dist_O2Sat[idx_obs_O2Sat]

            # Save information for this round
            date_str = row["Date Recorded"].strftime("%Y-%m-%d")
            AR.add_or_update_posterior(h, date_str, dist_AR)
            # Contains the same as AR.vmessages, but asn an array (not dict)
            # and the order is correct to reverse the speedup
            # I don't know if the dict entries would be sorted.
            AR_dist_given_M_matrix[n, :, h] = dist_AR

            log_p_D_given_M[n, h] = np.log(p_ecFEV1) + np.log(
                p_ecFEF2575
            )  # + np.log(p_O2Sat)

    # Do a backwards sweep to get the AR posteriors
    for n in range(N - 2, -1, -1):
        next_date = df_for_ID.loc[n + 1, "Date Recorded"]
        curr_date = df_for_ID.loc[n, "Date Recorded"]
        de = AR.calc_days_elapsed(curr_date, next_date)
        if de > 3:
            ValueError(f"Days elapsed is {de}, should be at most 3")

        for h, (HFEV1_obs, HO2Sat_obs) in enumerate(H_obs_list):
            next_AR = AR_dist_given_M_matrix[n + 1, :, h]
            next_AR_m = np.matmul(next_AR, AR.change_cpt[:, :, de - 1])
            next_AR_m = next_AR_m / next_AR_m.sum()
            curr_AR_posterior = AR_dist_given_M_matrix[n, :, h] * next_AR_m
            curr_AR_posterior = curr_AR_posterior / curr_AR_posterior.sum()
            AR_dist_given_M_matrix[n, :, h] = curr_AR_posterior

    toc = time.time()
    print(f"Time for {N} entries: {toc-tic:.2f} s")

    if debug:
        print("log(P(D|M)), first row", log_p_D_given_M[0, :])

    # For each HFEV1 model, given HFEV1_obs_list, we compute the log probability of the model given the data
    # log(P(M|D)) = 1/N * sum_n log(P(D|M)) + Cn_avg + log(P(M))
    log_p_M_given_D = np.zeros(H)
    for h, (HFEV1_obs, HO2Sat_obs) in enumerate(H_obs_list):
        log_p_M_hfev1 = np.log(HFEV1.cpt[HFEV1.get_bin_for_value(HFEV1_obs)[1]])
        log_p_M_ho2sat = np.log(HO2Sat.cpt[HO2Sat.get_bin_for_value(HO2Sat_obs)[1]])
        log_p_M_given_D[h] = (
            np.sum(log_p_D_given_M[:, h]) + log_p_M_hfev1 + log_p_M_ho2sat
        )

    # Exponentiating very negative numbers gives too small numbers
    # Setting the highest number to 1
    shift = 1 - log_p_M_given_D.max()
    log_p_M_given_D_shifted = log_p_M_given_D + shift

    # Exponentiate and normalise
    p_M_given_D = np.exp(log_p_M_given_D_shifted)
    p_M_given_D = p_M_given_D / p_M_given_D.sum()
    AR_dist_matrix = np.matmul(AR_dist_given_M_matrix, p_M_given_D)

    # Post process all AR dist matrices

    # Reshape P(M|D) into a 2D array for each HFEV1_obs, HO2Sat_obs
    p_M_given_D = p_M_given_D.reshape((len(HFEV1_obs_list), HO2Sat.card))

    # Fill the p(M|D) array with zeros on the left, where the HFEV1_obs < max ecFEV1
    n_impossible_hfev1_values = HFEV1.card - len(HFEV1_obs_list)
    p_M_given_D_full = np.vstack(
        [np.zeros((n_impossible_hfev1_values, HO2Sat.card)), p_M_given_D]
    )

    # Get the probability of HFEV1
    p_HFEV1_given_D = p_M_given_D_full.sum(axis=1)

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
    ih.plot_histogram(fig, HFEV1, p_HFEV1_given_D, 0, 6, 1, 1, annot=True)

    # Add heatmap with AR posteriors
    df1 = pd.DataFrame(
        data=AR_dist_matrix,
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

    title = f"{id} - Posterior HFEV1 after fusing all P(M_h|D)<br>AR prior: {ar_prior}<br>AR change CPT: {ar_change_cpt_suffix}"
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

    # return fig, p_M_given_D_full, p_M_given_D, AR_dist_given_M_matrix
    return


def process_id(inf_settings):
    print(f"Processing {inf_settings}")

    ar_prior, id = inf_settings
    ar_change_cpt_suffix = "_shift_span_[-20;20]_joint_sampling_3_days_model"

    if id == "101":
        dftmp = df[df["ID"] == "101"].iloc[:591]
    elif id == "405":
        dftmp = df[df["ID"] == "405"]
    elif id == "272":
        dftmp = df[df["ID"] == "272"].iloc[:417]
    elif id == "201":
        dftmp = df[df["ID"] == "201"].iloc[:289]
    elif id == "203":
        dftmp = df[df["ID"] == "203"].iloc[:285]

    return compute_log_p_D_given_M_per_HFEV1_HO2Sat_obs_temporal_AR(
        dftmp, ar_prior, ar_change_cpt_suffix, debug=False, save=True
    )


# Run the function in parallel using ProcessPoolExecutor
if __name__ == "__main__":

    interesting_ids = [
        # "132",
        # "146",
        # "177",
        # "180",
        # "202",
        # "527",
        # "117",
        # "131",
        # "134",
        # "191",
        # "139",
        # "253",
        "101",
        # Also from consec values
        "405",
        "272",
        "201",
        "203",
    ]

    ar_priors = [
        "uniform",
        "uniform message to HFEV1",
        "breathe (2 days model, ecFEV1, ecFEF25-75)",
    ]

    inf_settings = [
        list(zip([ar_prior] * len(interesting_ids), interesting_ids))
        for ar_prior in ar_priors
    ]
    inf_settings = list(itertools.chain(*inf_settings))

    # num_cores = os.cpu_count()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Map the function to the list of unique IDs
        list(executor.map(process_id, inf_settings))
