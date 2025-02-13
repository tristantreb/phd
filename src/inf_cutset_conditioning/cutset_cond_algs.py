import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import src.data.helpers as dh
import src.inf_cutset_conditioning.helpers as cutseth
import src.inference.helpers as ih
import src.models.builders as mb


def compute_log_p_D_given_M_for_noise_model(
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


def run_long_noise_model_through_time(
    df,
    ar_prior="uniform",
    ia_prior="uniform",
    ar_change_cpt_suffix=None,
    ecfev1_noise_model_suffix=None,
    fef2575_cpt_suffix=None,
    debug=False,
    save=False,
):
    inf_alg, HFEV1, HFEV1_obs_list, AR, ecFEV1, ecFEF2575prctecFEV1, model_spec_txt = (
        load_long_noise_model_through_time(
            df,
            ar_prior,
            ia_prior,
            ar_change_cpt_suffix,
            ecfev1_noise_model_suffix,
            fef2575_cpt_suffix,
        )
    )

    # if type == "fev1, fef2575":
    (
        fig,
        p_M_given_D,
        log_p_D_given_M,
        AR_given_M_and_D,
        AR_given_M_and_all_D,
        res_dict,
    ) = calc_log_p_D_given_M_and_AR_for_ID_any_obs(
        df,
        inf_alg,
        HFEV1,
        HFEV1_obs_list,
        AR,
        ecFEV1,
        ecFEF2575prctecFEV1,
        model_spec_txt,
        debug=debug,
        save=save,
    )
    # elif type == "fev1":
    #     (
    #         fig,
    #         p_M_given_D,
    #         log_p_D_given_M,
    #         AR_given_M_and_D,
    #         AR_given_M_and_all_D,
    #     ) = calc_log_p_D_given_M_and_AR_for_ID_obs_fev1(
    #         df,
    #         inf_alg,
    #         HFEV1,
    #         HFEV1_obs_list,
    #         AR,
    #         ecFEV1,
    #         model_spec_txt,
    #         debug=debug,
    #         save=save,
    #     )
    return (
        fig,
        p_M_given_D,
        log_p_D_given_M,
        AR_given_M_and_D,
        AR_given_M_and_all_D,
        res_dict,
    )


def run_long_noise_model_through_time_light(
    df, ar_prior="uniform", ia_prior="uniform", ar_change_cpt_suffix="", debug=False
):
    inf_alg, HFEV1, HFEV1_obs_list, AR, ecFEV1, ecFEF2575prctecFEV1, model_spec_txt = (
        load_long_noise_model_through_time_light(
            df, ar_prior, ia_prior, ar_change_cpt_suffix
        )
    )

    (fig, p_M_given_D, log_p_D_given_M, AR_given_M_and_D, AR_given_M_and_all_D) = (
        calc_log_p_D_given_M_and_AR_for_ID_obs_fev1_fef2575(
            df,
            inf_alg,
            HFEV1,
            HFEV1_obs_list,
            AR,
            ecFEV1,
            ecFEF2575prctecFEV1,
            model_spec_txt,
            debug=debug,
        )
    )
    return fig, p_M_given_D, AR_given_M_and_D


def load_long_noise_model_through_time(
    df,
    ar_prior="uniform",
    ia_prior="uniform",
    ar_change_cpt_suffix=None,
    ecfev1_noise_model_suffix=None,
    fef2575_cpt_suffix=None,
):
    height, age, sex = df.iloc[0][["Height", "Age", "Sex"]]

    # Initialize the noise model and its variables
    _, _, HFEV1, _, _, _, HO2Sat, *_ = (
        mb.o2sat_fev1_fef2575_long_model_noise_shared_healthy_vars_and_temporal_ar(
            height,
            age,
            sex,
            ar_change_cpt_suffix=ar_change_cpt_suffix,
            ecfev1_noise_model_suffix=ecfev1_noise_model_suffix,
            fef2575_cpt_suffix=fef2575_cpt_suffix,
        )
    )
    HFEV1_obs_idx_list = range(HFEV1.card)

    # Full inference model setup
    (
        _,
        inf_alg,
        HFEV1,
        uecFEV1,
        ecFEV1,
        AR,
        HO2Sat,
        O2SatFFA,
        IA,
        UO2Sat,
        O2Sat,
        ecFEF2575prctecFEV1,
        S,
    ) = mb.o2sat_fev1_fef2575_long_model_noise_shared_healthy_vars_and_temporal_ar(
        height,
        age,
        sex,
        ia_prior,
        ar_prior,
        ar_change_cpt_suffix,
        len(HFEV1_obs_idx_list),
        ecfev1_noise_model_suffix,
    )

    model_spec_txt = f"AR prior: {ar_prior}, ecFEV1 noise model {ecfev1_noise_model_suffix}<br>AR change CPT: {ar_change_cpt_suffix}"
    return (
        inf_alg,
        HFEV1,
        HFEV1_obs_idx_list,
        AR,
        ecFEV1,
        ecFEF2575prctecFEV1,
        model_spec_txt,
    )


def load_long_noise_model_through_time_light(
    df, ar_prior="uniform", ia_prior="uniform", ar_change_cpt_suffix=""
):
    height, age, sex = df.iloc[0][["Height", "Age", "Sex"]]

    # Initialize the noise model and its variables
    _, _, HFEV1, _, _, _, HO2Sat, *_ = (
        mb.o2sat_fev1_fef2575_long_model_noise_shared_healthy_vars_and_temporal_ar_light(
            height, age, sex, ar_change_cpt_suffix=ar_change_cpt_suffix
        )
    )
    # All bins are observed
    HFEV1_obs_idx_list = range(HFEV1.card)

    # Full inference model setup
    (
        _,
        inf_alg,
        HFEV1,
        uecFEV1,
        ecFEV1,
        AR,
        HO2Sat,
        O2SatFFA,
        IA,
        UO2Sat,
        O2Sat,
        ecFEF2575prctecFEV1,
        DE,
    ) = mb.o2sat_fev1_fef2575_long_model_noise_shared_healthy_vars_and_temporal_ar_light(
        height,
        age,
        sex,
        ia_prior,
        ar_prior,
        ar_change_cpt_suffix=ar_change_cpt_suffix,
        n_cutset_conditioned_states=len(HFEV1_obs_idx_list),
    )

    model_spec_txt = f"AR prior: {ar_prior}<br>AR change CPT: {ar_change_cpt_suffix}"
    return (
        inf_alg,
        HFEV1,
        HFEV1_obs_idx_list,
        AR,
        ecFEV1,
        ecFEF2575prctecFEV1,
        model_spec_txt,
    )


def calc_log_p_D_given_M_and_AR_for_ID_any_obs(
    df,
    inf_alg,
    HFEV1,
    HFEV1_obs_idx_list,
    AR,
    ecFEV1,
    ecFEF2575prctecFEV1,
    model_spec_txt,
    debug=False,
    save=False,
):
    df = df.copy().sort_values(by="Date Recorded").reset_index(drop=True)
    id = df.loc[0, "ID"]

    if debug:
        print(f"ID {id}")

    N = len(df)
    df = df.copy()
    H = len(HFEV1_obs_idx_list)
    log_p_D_given_M = np.zeros((N, H))
    res_dict = {}
    res_dict.update({"vevidence_ar": np.zeros((N, AR.card, H))})
    res_dict.update({"ecFEV1": np.zeros((N, ecFEV1.card, H))})
    res_dict.update({"ecFEF2575%ecFEV1": np.zeros((N, ecFEF2575prctecFEV1.card, H))})
    AR_given_M_and_past_D = np.zeros((N, AR.card, H))
    AR_given_M_and_same_day_D = np.zeros((N, AR.card, H))

    arr = np.ones(AR.card)
    arr /= arr.sum()
    uniform_from_o2_side = {
        "['O2 saturation if fully functional alveoli (%)', 'Healthy O2 saturation (%)', 'Airway resistance (%)'] -> Airway resistance (%)": arr
    }
    uniform_from_fef2575 = {
        "['ecFEF25-75 % ecFEV1 (%)', 'Airway resistance (%)'] -> Airway resistance (%)": arr
    }
    m_from_hfev1_key = "Healthy FEV1 (L) -> ['Underlying ecFEV1 (L)', 'Healthy FEV1 (L)', 'Airway resistance (%)']"
    m_from_hfev1_dict = {}
    m_from_fev_factor_key = "['Underlying ecFEV1 (L)', 'Healthy FEV1 (L)', 'Airway resistance (%)'] -> Airway resistance (%)"
    m_from_fev_factor_dict = {}

    # Get the joint probability of ecFEV1 and ecFEF2575 given the model for this individual
    # Process each row
    # P(model | data) prop_to P(data | model) * P(model)
    # P(data | model) = P(ecFEV1, ecFEF2575 | HFEV1) = P(ecFEV1 | HFEV1) * P( ecFEF2575 | HFEV1, ecFEV1)
    tic = time.time()
    for n, row in df.iterrows():
        curr_date = df.loc[n, "Date Recorded"]
        # There is no prev day if it's the first day
        prev_date = df.iloc[n - 1]["Date Recorded"] if n > 0 else None
        if debug:
            print(f"Row {n + 1}/{N}, Date: {curr_date}, Prev Date: {prev_date}")

        # For each model given an HFEV1 observation
        for h, HFEV1_bin_idx in enumerate(HFEV1_obs_idx_list):
            vevidence_ar = cutseth.build_vevidence_cutset_conditioned_ar(
                AR, h, curr_date, prev_date, next_date=None, debug=debug
            )
            res_dict["vevidence_ar"][n, :, h] = vevidence_ar.values

            if debug:
                print(
                    f"HFEV1_obs_idx: {HFEV1_bin_idx}, vevidence_ar: {vevidence_ar.values}"
                )

            if not np.isnan(row["idx ecFEV1 (L)"]) and not np.isnan(
                row["idx ecFEF25-75 % ecFEV1 (%)"]
            ):
                if debug:
                    print("Both ecFEV1 and ecFEF25-75 observed")
                (
                    log_p_D_given_M_for_row,
                    dist_AR,
                    dist_ecFEV1,
                    dist_ecFEF2575prctecFEV1,
                ) = get_AR_and_p_log_D_given_M_obs_fev1_and_fef2575(
                    row,
                    inf_alg,
                    HFEV1,
                    HFEV1_bin_idx,
                    ecFEV1,
                    ecFEF2575prctecFEV1,
                    AR,
                    vevidence_ar,
                    uniform_from_o2_side | uniform_from_fef2575,
                    m_from_hfev1_dict,
                    m_from_hfev1_key,
                    m_from_fev_factor_dict,
                    m_from_fev_factor_key,
                )
            elif not np.isnan(row["idx ecFEV1 (L)"]) and np.isnan(
                row["idx ecFEF25-75 % ecFEV1 (%)"]
            ):
                if debug:
                    print("Only ecFEV1 observed")
                log_p_D_given_M_for_row, dist_AR, dist_ecFEV1 = (
                    get_AR_and_p_log_D_given_M_obs_fev1(
                        row,
                        inf_alg,
                        HFEV1,
                        HFEV1_bin_idx,
                        ecFEV1,
                        AR,
                        vevidence_ar,
                        uniform_from_o2_side | uniform_from_fef2575,
                        m_from_hfev1_dict,
                        m_from_hfev1_key,
                        m_from_fev_factor_dict,
                        m_from_fev_factor_key,
                    )
                )
                dist_ecFEF2575prctecFEV1 = np.zeros(ecFEF2575prctecFEV1.card)
            elif np.isnan(row["idx ecFEV1 (L)"]) and np.isnan(
                row["idx ecFEF25-75 % ecFEV1 (%)"]
            ):
                if debug:
                    print("No data observed")
                log_p_D_given_M_for_row, dist_AR = get_AR_and_p_log_D_given_M_no_obs(
                    inf_alg,
                    HFEV1,
                    HFEV1_bin_idx,
                    AR,
                    vevidence_ar,
                    uniform_from_o2_side | uniform_from_fef2575,
                    m_from_hfev1_dict,
                    m_from_hfev1_key,
                )
                dist_ecFEV1 = np.zeros(ecFEV1.card)
                dist_ecFEF2575prctecFEV1 = np.zeros(ecFEF2575prctecFEV1.card)
            else:
                raise ValueError(
                    f"Unexpected combination of observed variables for row\n{row}"
                )

            log_p_D_given_M[n, h] = log_p_D_given_M_for_row
            AR_given_M_and_past_D[n, :, h] = dist_AR
            res_dict["ecFEV1"][n, :, h] = dist_ecFEV1
            res_dict["ecFEF2575%ecFEV1"][n, :, h] = dist_ecFEF2575prctecFEV1

            # Add the AR dist to be used for as next day's AR vevidence
            date_str = row["Date Recorded"].strftime("%Y-%m-%d")
            AR.add_or_update_posterior(h, date_str, dist_AR, debug)

            # Get the AR without the contribution from the past days' data
            # This will be used to get the contribution from the next days' data
            # Check that wherever vevidence_ar is 0, the res2[AR.name] is also 0
            AR_without_vevidence = np.divide(
                dist_AR, vevidence_ar.values, where=vevidence_ar.values != 0
            )
            AR_without_vevidence /= AR_without_vevidence.sum()
            AR_given_M_and_same_day_D[n, :, h] = AR_without_vevidence

    toc = time.time()
    print(f"{id} - Time for {N} entries: {toc-tic:.2f} s")

    AR_given_M_and_all_D = run_backward_sweep_to_get_ar_posteriors(
        df,
        N,
        H,
        HFEV1_obs_idx_list,
        AR,
        AR_given_M_and_past_D,
        AR_given_M_and_same_day_D,
        debug,
    )

    p_M_given_D, log_p_D_given_M, AR_given_M_and_D, AR_given_M_and_all_D = (
        fuse_results_from_conditioned_models(
            HFEV1, H, HFEV1_obs_idx_list, log_p_D_given_M, AR_given_M_and_all_D
        )
    )

    fig = plot_cutset_cond_results(
        df, HFEV1, p_M_given_D, AR, AR_given_M_and_D, model_spec_txt, save
    )

    return (
        fig,
        p_M_given_D,
        log_p_D_given_M,
        AR_given_M_and_D,
        AR_given_M_and_all_D,
        res_dict,
    )


def calc_log_p_D_given_M_and_AR_for_ID_obs_fev1_fef2575(
    df,
    inf_alg,
    HFEV1,
    HFEV1_obs_idx_list,
    AR,
    ecFEV1,
    ecFEF2575prctecFEV1,
    model_spec_txt,
    debug=False,
    save=False,
):
    df = df.copy().sort_values(by="Date Recorded").reset_index(drop=True)
    id = df.loc[0, "ID"]

    if debug:
        print(f"ID {id}")

    N = len(df)
    df = df.copy()
    H = len(HFEV1_obs_idx_list)
    log_p_D_given_M = np.zeros((N, H))
    AR_given_M_and_past_D = np.zeros((N, AR.card, H))
    AR_given_M_and_same_day_D = np.zeros((N, AR.card, H))

    arr = np.ones(AR.card)
    arr /= arr.sum()
    uniform_from_o2_side = {
        "['O2 saturation if fully functional alveoli (%)', 'Healthy O2 saturation (%)', 'Airway resistance (%)'] -> Airway resistance (%)": arr
    }
    uniform_from_fef2575 = {
        "['ecFEF25-75 % ecFEV1 (%)', 'Airway resistance (%)'] -> Airway resistance (%)": arr
    }
    m_from_hfev1_key = "Healthy FEV1 (L) -> ['Underlying ecFEV1 (L)', 'Healthy FEV1 (L)', 'Airway resistance (%)']"
    m_from_hfev1_dict = {}
    m_from_fev_factor_key = "['Underlying ecFEV1 (L)', 'Healthy FEV1 (L)', 'Airway resistance (%)'] -> Airway resistance (%)"
    m_from_fev_factor_dict = {}

    # Get the joint probability of ecFEV1 and ecFEF2575 given the model for this individual
    # Process each row
    # P(model | data) prop_to P(data | model) * P(model)
    # P(data | model) = P(ecFEV1, ecFEF2575 | HFEV1) = P(ecFEV1 | HFEV1) * P( ecFEF2575 | HFEV1, ecFEV1)
    tic = time.time()
    for n, row in df.iterrows():
        curr_date = df.loc[n, "Date Recorded"]
        # There is no prev day if it's the first day
        prev_date = df.iloc[n - 1]["Date Recorded"] if n > 0 else None
        if debug:
            print(f"Row {n + 1}/{N}, Date: {curr_date}, Prev Date: {prev_date}")

        # For each model given an HFEV1 observation
        for h, HFEV1_bin_idx in enumerate(HFEV1_obs_idx_list):
            vevidence_ar = cutseth.build_vevidence_cutset_conditioned_ar(
                AR, h, curr_date, prev_date, next_date=None, debug=debug
            )

            if debug:
                print(
                    f"HFEV1_obs: {HFEV1_bin_idx}, vevidence_ar: {vevidence_ar.values}"
                )

            # Perform inference
            (
                log_p_ecfev1_fef2575_given_M,
                dist_AR,
                dist_ecFEV1,
                dist_ecFEF2575prctecFEV1,
            ) = get_AR_and_p_log_D_given_M_obs_fev1_and_fef2575(
                row,
                inf_alg,
                HFEV1,
                HFEV1_bin_idx,
                ecFEV1,
                ecFEF2575prctecFEV1,
                AR,
                vevidence_ar,
                uniform_from_o2_side | uniform_from_fef2575,
                m_from_hfev1_dict,
                m_from_hfev1_key,
                m_from_fev_factor_dict,
                m_from_fev_factor_key,
            )
            log_p_D_given_M[n, h] = log_p_ecfev1_fef2575_given_M
            AR_given_M_and_past_D[n, :, h] = dist_AR

            # Add the AR dist to be used for as next day's AR vevidence
            date_str = row["Date Recorded"].strftime("%Y-%m-%d")
            AR.add_or_update_posterior(h, date_str, dist_AR, debug)

            # Get the AR without the contribution from the past days' data
            # This will be used to get the contribution from the next days' data
            # Check that wherever vevidence_ar is 0, the res2[AR.name] is also 0
            AR_without_vevidence = np.divide(
                dist_AR, vevidence_ar.values, where=vevidence_ar.values != 0
            )
            AR_without_vevidence /= AR_without_vevidence.sum()
            AR_given_M_and_same_day_D[n, :, h] = AR_without_vevidence

    toc = time.time()
    print(f"{id} - Time for {N} entries: {toc-tic:.2f} s")

    AR_given_M_and_all_D = run_backward_sweep_to_get_ar_posteriors(
        df,
        N,
        H,
        HFEV1_obs_idx_list,
        AR,
        AR_given_M_and_past_D,
        AR_given_M_and_same_day_D,
        debug,
    )

    p_M_given_D, log_p_D_given_M, AR_given_M_and_D, AR_given_M_and_all_D = (
        fuse_results_from_conditioned_models(
            HFEV1, H, HFEV1_obs_idx_list, log_p_D_given_M, AR_given_M_and_all_D
        )
    )

    fig = plot_cutset_cond_results(
        df, HFEV1, p_M_given_D, AR, AR_given_M_and_D, model_spec_txt, save
    )

    return (fig, p_M_given_D, log_p_D_given_M, AR_given_M_and_D, AR_given_M_and_all_D)


def calc_log_p_D_given_M_and_AR_for_ID_obs_fev1(
    df,
    inf_alg,
    HFEV1,
    HFEV1_obs_idx_list,
    AR,
    ecFEV1,
    model_spec_txt,
    debug=False,
    save=False,
):
    df = df.copy().sort_values(by="Date Recorded").reset_index(drop=True)
    id = df.loc[0, "ID"]

    if debug:
        print(f"ID {id}")

    N = len(df)
    df = df.copy()
    H = len(HFEV1_obs_idx_list)
    log_p_D_given_M = np.zeros((N, H))
    AR_given_M_and_past_D = np.zeros((N, AR.card, H))
    AR_given_M_and_same_day_D = np.zeros((N, AR.card, H))

    arr = np.ones(AR.card)
    arr /= arr.sum()
    uniform_from_o2_side = {
        "['O2 saturation if fully functional alveoli (%)', 'Healthy O2 saturation (%)', 'Airway resistance (%)'] -> Airway resistance (%)": arr
    }
    uniform_from_fef2575 = {
        "['ecFEF25-75 % ecFEV1 (%)', 'Airway resistance (%)'] -> Airway resistance (%)": arr
    }
    m_from_hfev1_key = "Healthy FEV1 (L) -> ['Underlying ecFEV1 (L)', 'Healthy FEV1 (L)', 'Airway resistance (%)']"
    m_from_hfev1_dict = {}
    m_from_fev_factor_key = "['Underlying ecFEV1 (L)', 'Healthy FEV1 (L)', 'Airway resistance (%)'] -> Airway resistance (%)"
    m_from_fev_factor_dict = {}

    # Get the joint probability of ecFEV1 and ecFEF2575 given the model for this individual
    # Process each row
    # P(model | data) prop_to P(data | model) * P(model)
    # P(data | model) = P(ecFEV1, ecFEF2575 | HFEV1) = P(ecFEV1 | HFEV1) * P( ecFEF2575 | HFEV1, ecFEV1)
    tic = time.time()
    for n, row in df.iterrows():
        curr_date = df.loc[n, "Date Recorded"]
        # There is no prev day if it's the first day
        prev_date = df.iloc[n - 1]["Date Recorded"] if n > 0 else None
        if debug:
            print(f"Row {n + 1}/{N}, Date: {curr_date}, Prev Date: {prev_date}")

        # For each model given an HFEV1 observation
        for h, HFEV1_bin_idx in enumerate(HFEV1_obs_idx_list):
            vevidence_ar = cutseth.build_vevidence_cutset_conditioned_ar(
                AR, h, curr_date, prev_date, next_date=None, debug=debug
            )

            if debug:
                print(
                    f"HFEV1_obs: {HFEV1_bin_idx}, vevidence_ar: {vevidence_ar.values}"
                )

            # Perform inference
            p_log_fev1_given_M, dist_AR = get_AR_and_p_log_D_given_M_obs_fev1(
                row,
                inf_alg,
                HFEV1,
                HFEV1_bin_idx,
                ecFEV1,
                AR,
                vevidence_ar,
                uniform_from_o2_side | uniform_from_fef2575,
                m_from_hfev1_dict,
                m_from_hfev1_key,
                m_from_fev_factor_dict,
                m_from_fev_factor_key,
            )

            log_p_D_given_M[n, h] = p_log_fev1_given_M
            AR_given_M_and_past_D[n, :, h] = dist_AR

            # Add the AR dist to be used for as next day's AR vevidence
            date_str = row["Date Recorded"].strftime("%Y-%m-%d")
            AR.add_or_update_posterior(h, date_str, dist_AR, debug)

            # Get the AR without the contribution from the past days' data
            # This will be used to get the contribution from the next days' data
            # Check that wherever vevidence_ar is 0, the res2[AR.name] is also 0
            AR_without_vevidence = np.divide(
                dist_AR, vevidence_ar.values, where=vevidence_ar.values != 0
            )
            AR_without_vevidence /= AR_without_vevidence.sum()
            AR_given_M_and_same_day_D[n, :, h] = AR_without_vevidence

    toc = time.time()
    print(f"{id} - Time for {N} entries: {toc-tic:.2f} s")

    AR_given_M_and_all_D = run_backward_sweep_to_get_ar_posteriors(
        df,
        N,
        H,
        HFEV1_obs_idx_list,
        AR,
        AR_given_M_and_past_D,
        AR_given_M_and_same_day_D,
        debug,
    )

    p_M_given_D, log_p_D_given_M, AR_given_M_and_D, AR_given_M_and_all_D = (
        fuse_results_from_conditioned_models(
            HFEV1, H, HFEV1_obs_idx_list, log_p_D_given_M, AR_given_M_and_all_D
        )
    )

    fig = plot_cutset_cond_results(
        df, HFEV1, p_M_given_D, AR, AR_given_M_and_D, model_spec_txt, save
    )

    return (fig, p_M_given_D, log_p_D_given_M, AR_given_M_and_D, AR_given_M_and_all_D)


def get_AR_and_p_log_D_given_M_obs_fev1_and_fef2575(
    data,
    bp,
    HFEV1,
    HFEV1_bin_idx,
    ecFEV1,
    ecFEF2575prctecFEV1,
    AR,
    vevidence_ar,
    precomp_messages,
    m_from_hfev1_dict,
    m_from_hfev1_key,
    m_from_fev_factor_dict,
    m_from_fev_factor_key,
):

    # Get P(ecFEV1 | model conditionned on HFEV1_obs)
    precomp_messages1 = precomp_messages
    ref = f"{HFEV1_bin_idx}"
    add_ref = True
    if ref in m_from_hfev1_dict:
        precomp_messages1.update(m_from_hfev1_dict[ref])
        add_ref = False
    res1, messages = bp.query(
        variables=[ecFEV1.name],
        evidence={HFEV1.name: HFEV1_bin_idx},
        virtual_evidence=[vevidence_ar],
        precomp_messages=precomp_messages1,
        get_messages=True,
    )
    if add_ref:
        m_from_hfev1_dict.update({ref: {m_from_hfev1_key: messages[m_from_hfev1_key]}})
    dist_ecFEV1 = res1[ecFEV1.name].values
    p_ecFEV1 = dist_ecFEV1[data["idx ecFEV1 (L)"]]
    log_p_D_given_M = np.log(p_ecFEV1)

    # Get P(ecFEF2575 | model conditionned on HFEV1_obs, ecFEV1)
    precomp_messages2 = precomp_messages
    ref = f"{HFEV1_bin_idx}_{data['idx ecFEV1 (L)']}"
    add_ref = True
    if ref in m_from_fev_factor_dict:
        precomp_messages2.update(m_from_fev_factor_dict[ref])
        add_ref = False
    res2, messages = bp.query(
        variables=[ecFEF2575prctecFEV1.name, AR.name],
        evidence={
            HFEV1.name: HFEV1_bin_idx,
            ecFEV1.name: data["idx ecFEV1 (L)"],
        },
        virtual_evidence=[vevidence_ar],
        precomp_messages=precomp_messages2,
        get_messages=True,
    )
    if add_ref:
        m_from_fev_factor_dict.update(
            {ref: {m_from_fev_factor_key: messages[m_from_fev_factor_key]}}
        )
    dist_ecFEF2575prctecFEV1 = res2[ecFEF2575prctecFEV1.name].values
    p_ecFEF2575 = dist_ecFEF2575prctecFEV1[data["idx ecFEF2575%ecFEV1"]]
    log_p_D_given_M += np.log(p_ecFEF2575)

    # Add FEF25-75 observation to P(AR|HFEV1, ecFEV1)
    # Done manually for time efficiency
    m_to_factor = np.zeros(ecFEF2575prctecFEV1.card)
    m_to_factor[data["idx ecFEF2575%ecFEV1"]] = 1
    factor_to_AR = np.matmul(m_to_factor, ecFEF2575prctecFEV1.cpt)
    factor_to_AR /= factor_to_AR.sum()
    dist_AR = res2[AR.name].values * factor_to_AR
    dist_AR /= dist_AR.sum()

    return log_p_D_given_M, dist_AR, dist_ecFEV1, dist_ecFEF2575prctecFEV1


def get_AR_and_p_log_D_given_M_obs_fev1(
    data,
    bp,
    HFEV1,
    HFEV1_bin_idx,
    ecFEV1,
    AR,
    vevidence_ar,
    precomp_messages,
    m_from_hfev1_dict,
    m_from_hfev1_key,
    m_from_fev_factor_dict,
    m_from_fev_factor_key,
):

    # Get P(ecFEV1 | model conditionned on HFEV1_obs)
    precomp_messages1 = precomp_messages
    ref = f"{HFEV1_bin_idx}"
    add_ref = True
    if ref in m_from_hfev1_dict:
        precomp_messages.update(m_from_hfev1_dict[ref])
        add_ref = False
    res1, messages = bp.query(
        variables=[ecFEV1.name],
        evidence={HFEV1.name: HFEV1_bin_idx},
        virtual_evidence=[vevidence_ar],
        precomp_messages=precomp_messages1,
        get_messages=True,
    )
    if add_ref:
        m_from_hfev1_dict.update({ref: {m_from_hfev1_key: messages[m_from_hfev1_key]}})
    dist_ecFEV1 = res1[ecFEV1.name].values
    p_ecFEV1 = dist_ecFEV1[data["idx ecFEV1 (L)"]]
    log_p_D_given_M = np.log(p_ecFEV1)

    # Get AR
    precomp_messages2 = precomp_messages
    ref = f"{HFEV1_bin_idx}_{data['idx ecFEV1 (L)']}"
    add_ref = True
    if ref in m_from_fev_factor_dict:
        precomp_messages.update(m_from_fev_factor_dict[ref])
        add_ref = False
    res2, messages = bp.query(
        variables=[AR.name],
        evidence={
            HFEV1.name: HFEV1_bin_idx,
            ecFEV1.name: data["idx ecFEV1 (L)"],
        },
        virtual_evidence=[vevidence_ar],
        precomp_messages=precomp_messages2,
        get_messages=True,
    )
    if add_ref:
        m_from_fev_factor_dict.update(
            {ref: {m_from_fev_factor_key: messages[m_from_fev_factor_key]}}
        )

    dist_AR = res2[AR.name].values
    dist_AR /= dist_AR.sum()
    return log_p_D_given_M, dist_AR, dist_ecFEV1


def get_AR_and_p_log_D_given_M_no_obs(
    bp,
    HFEV1,
    HFEV1_bin_idx,
    AR,
    vevidence_ar,
    precomp_messages,
    m_from_hfev1_dict,
    m_from_hfev1_key,
):
    # Get P(AR | model conditionned on HFEV1_obs)
    ref = f"{HFEV1_bin_idx}"
    add_ref = True
    if ref in m_from_hfev1_dict:
        precomp_messages.update(m_from_hfev1_dict[ref])
        add_ref = False
    res, messages = bp.query(
        variables=[AR.name],
        evidence={HFEV1.name: HFEV1_bin_idx},
        virtual_evidence=[vevidence_ar],
        precomp_messages=precomp_messages,
        get_messages=True,
    )
    if add_ref:
        m_from_hfev1_dict.update({ref: {m_from_hfev1_key: messages[m_from_hfev1_key]}})

    dist_AR = res[AR.name].values
    dist_AR /= dist_AR.sum()

    # No contribution from data on this day
    log_D_given_M = np.nan
    return log_D_given_M, dist_AR


def run_backward_sweep_to_get_ar_posteriors(
    df,
    N,
    H,
    HFEV1_obs_idx_list,
    AR,
    AR_given_M_and_past_D,
    AR_given_M_and_same_day_D,
    debug,
):
    AR_given_M_and_future_D = np.zeros((N, AR.card, H))
    AR_given_M_and_all_D = np.zeros((N, AR.card, H))

    # Do a backwards sweep to get the AR posteriors
    AR_given_M_and_future_D = AR_given_M_and_same_day_D.copy()

    # The AR posterior on the last day only has contribution from past data
    AR_given_M_and_all_D[N - 1, :, :] = AR_given_M_and_past_D[N - 1, :, :]

    # Hence we can start computing the AR posteriors from the second last day
    for n in range(N - 2, -1, -1):
        next_date = df.loc[n + 1, "Date Recorded"]
        curr_date = df.loc[n, "Date Recorded"]
        de = AR.calc_days_elapsed(curr_date, next_date)
        if debug:
            print(
                f"Propagating AR backwards: Processing idx {n}/{N-1}, curr date {curr_date}, next date {next_date}, days elapsed {de}"
            )
        if de > 3:
            ValueError(f"Days elapsed is {de}, should be at most 3")

        for h, HFEV1_bin_idx in enumerate(HFEV1_obs_idx_list):
            next_AR = AR_given_M_and_future_D[n + 1, :, h]
            curr_AR = AR_given_M_and_future_D[n, :, h]
            past_AR = AR_given_M_and_past_D[n, :, h]

            # Compute contribution from future days' data
            next_AR_m = np.matmul(next_AR, AR.change_cpt[:, :, de - 1])
            next_AR_m = next_AR_m / next_AR_m.sum()

            # Compute posterior for past and future data

            curr_AR_posterior = past_AR * next_AR_m
            curr_AR_posterior = curr_AR_posterior / curr_AR_posterior.sum()
            AR_given_M_and_all_D[n, :, h] = curr_AR_posterior

            # Include future data in current day
            curr_AR_with_future_D = curr_AR * next_AR_m
            curr_AR_with_future_D = curr_AR_with_future_D / curr_AR_with_future_D.sum()
            AR_given_M_and_future_D[n, :, h] = curr_AR_with_future_D

    return AR_given_M_and_all_D


def fuse_results_from_conditioned_models(
    HFEV1, H, HFEV1_obs_idx_list, log_p_D_given_M, AR_given_M_and_all_D
):
    # For each HFEV1 model, given HFEV1_obs_idx_list, we compute the log probability of the model given the data
    # log(P(M|D)) = 1/N * sum_n log(P(D|M)) + Cn_avg + log(P(M))
    log_p_M_given_D = np.zeros(H)
    for h, HFEV1_bin_idx in enumerate(HFEV1_obs_idx_list):
        log_p_M_hfev1 = np.log(HFEV1.cpt[HFEV1_bin_idx])
        # Do nan sums to remove contributions from days without data
        log_p_M_given_D[h] = np.nansum(log_p_D_given_M[:, h]) + log_p_M_hfev1

    # Exponentiating very negative numbers gives too small numbers
    # Setting the highest number to 1 to prevent numerical issues
    log_p_M_given_D -= log_p_M_given_D.max()
    p_M_given_D = np.exp(log_p_M_given_D)
    p_M_given_D /= p_M_given_D.sum()
    AR_given_M_and_D = np.matmul(AR_given_M_and_all_D, p_M_given_D)
    return p_M_given_D, log_p_D_given_M, AR_given_M_and_D, AR_given_M_and_all_D


def plot_cutset_cond_results(
    df, HFEV1, p_HFEV1_given_D, AR, AR_given_M_and_D, model_spec_txt, save
):
    id = df.loc[0, "ID"]
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
        data=AR_given_M_and_D,
        columns=AR.get_bins_str(),
        index=df["Date Recorded"].apply(lambda date: date.strftime("%Y-%m-%d")),
    )
    colorscale = [
        [0, "white"],
        [0.01, "red"],
        [0.05, "yellow"],
        [0.1, "cyan"],
        [0.6, "blue"],
        [1, "black"],
    ]
    # colorscale = [
    #     # [0, "white"],
    #     # [0.01, "red"],
    #     # [0.05, "yellow"],
    #     [0, "cyan"],
    #     [0.5, "blue"],
    #     [1, "black"],
    # ]

    fig.add_trace(
        go.Heatmap(z=df1.T, x=df1.index, y=df1.columns, coloraxis="coloraxis1"),
        row=2,
        col=1,
    )

    title = f"{id} - Results after fusing all P(M_h|D) {len(df)} entries<br>{model_spec_txt}"
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
    fig.update_layout(title_font_size=14)

    # Add Date on x axis
    fig.update_xaxes(title_text=HFEV1.name, row=1, col=1)
    fig.update_yaxes(title_text="p", row=1, col=1)
    fig.update_yaxes(title_text=AR.name, row=2, col=1)
    fig.update_xaxes(
        title_text="Date (categorical)",
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
    return fig
