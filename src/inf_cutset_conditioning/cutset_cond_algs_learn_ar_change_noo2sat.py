import itertools
import time

import numpy as np

import inf_cutset_conditioning.cutset_cond_algs as cca
import inf_cutset_conditioning.helpers as cutseth
import models.builders as mb
import models.helpers as mh

"""
Updates wrt to src/inf_cutset_conditioning/cutset_cond_long_noise_model_through_time.py:
- Removed o2 sat side form the graph in the loading function (not for the light model though)
- Removed the precomp messages for the o2sat side

It solved the issue with the discretisation. Hence, not using double integrals might not have been the issue,
it probably comes from an error in how I implemented precomputed messages - but I validated that...
"""


def run_long_noise_model_through_time(
    df,
    ar_prior="uniform",
    ar_change_cpt_suffix=None,
    ecfev1_noise_model_suffix=None,
    fef2575_cpt_suffix=None,
    n_days_consec=3,
    light=False,
    get_p_s_given_d=False,
    debug=False,
    save=False,
):
    (
        inf_alg,
        HFEV1,
        h_s_obs_states,
        AR,
        ecFEV1,
        ecFEF2575prctecFEV1,
        model_spec_txt,
        S,
        cond_hfev1_card,
    ) = load_long_noise_model_through_time(
        df,
        ar_prior,
        ar_change_cpt_suffix,
        ecfev1_noise_model_suffix,
        fef2575_cpt_suffix,
        light,
    )

    (log_p_D_given_M, res_dict, AR_given_M_and_past_D, AR_given_M_and_same_day_D) = (
        calc_log_p_D_given_M_and_AR_for_ID_any_obs(
            df,
            inf_alg,
            HFEV1,
            h_s_obs_states,
            AR,
            ecFEV1,
            ecFEF2575prctecFEV1,
            model_spec_txt,
            S,
            n_days_consec,
            cond_hfev1_card,
            debug=debug,
            save=save,
        )
    )

    if get_p_s_given_d:
        log_p_S_given_D = fuse_results_to_compute_P_S_given_D(
            id, HFEV1, S, log_p_D_given_M
        )
        return (
            log_p_S_given_D,
            res_dict,
        )

    else:
        AR_given_M_and_all_D = run_backward_sweep_to_get_ar_posteriors(
            df,
            h_s_obs_states,
            AR,
            AR_given_M_and_past_D,
            AR_given_M_and_same_day_D,
            debug,
        )

        p_M_given_D, AR_given_M_and_D, AR_given_M_and_all_D = (
            fuse_results_from_conditioned_models(
                HFEV1, h_s_obs_states, log_p_D_given_M, AR_given_M_and_all_D
            )
        )

        # Put p_M_given_D in 2 dimensions, one for hfev1, one for s
        p_M_given_D = p_M_given_D.reshape((cond_hfev1_card, S.card))
        AR_given_M_and_all_D = AR_given_M_and_all_D.reshape(
            (len(df), AR.card, cond_hfev1_card, S.card)
        )
        for S_obs_idx in range(S.card):
            model_spec_txt_for_S = f"{model_spec_txt}, S: {S_obs_idx}"

            p_HFEV1_given_D = p_M_given_D[:, S_obs_idx]
            p_HFEV1_given_D /= p_HFEV1_given_D.sum()

            AR_given_HFEV1_and_D = AR_given_M_and_all_D[:, :, :, S_obs_idx]
            AR_given_HFEV1_and_D = np.matmul(AR_given_HFEV1_and_D, p_HFEV1_given_D)

            # Add HFEV1.card - hfev1_card zeros to p_HFEV1_given_D
            p_HFEV1_given_D = np.concatenate(
                [np.zeros(HFEV1.card - cond_hfev1_card), p_HFEV1_given_D]
            )

            # p_M_given_D has HFEV1.card
            # AR_given_M_and_D has N x AR.card
            fig = cca.plot_short_term_long_model_cutset_cond_results(
                df,
                HFEV1,
                p_HFEV1_given_D,
                AR,
                AR_given_HFEV1_and_D,
                model_spec_txt_for_S,
                save,
            )
            # fig = cca.plot_cutset_cond_results(
            #     df,
            #     HFEV1,
            #     p_HFEV1_given_D,
            #     AR,
            #     AR_given_HFEV1_and_D,
            #     model_spec_txt_for_S,
            #     save,
            # )

        return (
            fig,
            p_M_given_D,
            p_HFEV1_given_D,
            log_p_D_given_M,
            AR_given_M_and_D,
            AR_given_M_and_all_D,
            res_dict,
        )


def load_long_noise_model_through_time(
    df,
    ar_prior="uniform",
    ar_change_cpt_suffix=None,
    ecfev1_noise_model_suffix=None,
    fef2575_cpt_suffix=None,
    light=False,
):
    height, age, sex = df.iloc[0][["Height", "Age", "Sex"]]

    (
        _,
        _,
        HFEV1,
        uFEV1,
        ecFEV1,
        AR,
        _,
        S,
    ) = mb.fev1_fef2575_long_model_noise_shared_healthy_vars_and_temporal_ar(
        height,
        age,
        sex,
        ar_change_cpt_suffix=ar_change_cpt_suffix,
        ecfev1_noise_model_suffix=ecfev1_noise_model_suffix,
        fef2575_cpt_suffix=fef2575_cpt_suffix,
        light=light,
    )

    def get_min_possible_HFEV1_given_max_FEV1():
        max_ecfev1 = np.zeros(ecFEV1.card)
        max_ecfev1[df["idx ecFEV1 (L)"].max()] = 1
        # Compute underling ecFEV1 given observed max ecFEV1 = add noise to max ecFEV1
        ufev1 = np.matmul(max_ecfev1, ecFEV1.cpt)
        argmin_ufev1 = np.nonzero(ufev1)[0][0]
        min_ufev1 = uFEV1.midbins[argmin_ufev1]
        argmin_hfev1 = HFEV1.get_bin_idx_for_value(min_ufev1)
        # This code is working
        # print(f"ufev1: {ufev1}\nargmin_ufev1: {argmin_ufev1}\nmin_ufev1: {min_ufev1}\nargmin_hfev1: {argmin_hfev1}\nHFEV1 midbin: {HFEV1.midbins[argmin_hfev1]}")
        return argmin_hfev1

    min_possible_hfev1_under_model = get_min_possible_HFEV1_given_max_FEV1()
    if min_possible_hfev1_under_model > 0:
        print(
            f"Warning - min_possible_hfev1_under_model: {min_possible_hfev1_under_model}"
        )
    HFEV1_obs_idx_list = range(min_possible_hfev1_under_model, HFEV1.card)
    # HFEV1_obs_idx_list = [40]
    cond_hfev1_card = len(HFEV1_obs_idx_list)

    S_obs_idx_list = range(S.card)
    h_s_obs_states = list(itertools.product(HFEV1_obs_idx_list, S_obs_idx_list))
    # print(f"h_s_obs_states: {h_s_obs_states}")

    (
        _,
        inf_alg,
        HFEV1,
        uFEV1,
        ecFEV1,
        AR,
        ecFEF2575prctecFEV1,
        S,
    ) = mb.fev1_fef2575_long_model_noise_shared_healthy_vars_and_temporal_ar(
        height,
        age,
        sex,
        ar_prior,
        ar_change_cpt_suffix,
        len(h_s_obs_states),
        ecfev1_noise_model_suffix,
        fef2575_cpt_suffix,
        light,
    )

    model_spec_txt = f"AR prior: {ar_prior}, ecFEV1 noise model {ecfev1_noise_model_suffix}<br>AR change CPT: {ar_change_cpt_suffix}"
    return (
        inf_alg,
        HFEV1,
        h_s_obs_states,
        AR,
        ecFEV1,
        ecFEF2575prctecFEV1,
        model_spec_txt,
        S,
        cond_hfev1_card,
    )


def calc_log_p_D_given_M_and_AR_for_ID_any_obs(
    df,
    inf_alg,
    HFEV1,
    h_s_obs_states,
    AR,
    ecFEV1,
    ecFEF2575prctecFEV1,
    model_spec_txt,
    S,
    n_days_consec,
    cond_hfev1_card,
    debug=False,
    save=False,
):
    df = df.copy().sort_values(by="Date Recorded").reset_index(drop=True)
    id = df.loc[0, "ID"]

    if debug:
        print(f"ID {id}")

    N = len(df)
    df = df.copy()
    H = len(h_s_obs_states)
    log_p_D_given_M = np.zeros((N, H))
    res_dict = {}
    res_dict.update({"vevidence_ar": np.zeros((N, AR.card, H))})
    res_dict.update({"ecFEV1": np.zeros((N, ecFEV1.card, H))})
    res_dict.update({"ecFEF2575%ecFEV1": np.zeros((N, ecFEF2575prctecFEV1.card, H))})
    AR_given_M_and_past_D = np.zeros((N, AR.card, H))
    AR_given_M_and_same_day_D = np.zeros((N, AR.card, H))

    arr = np.ones(AR.card)
    arr /= arr.sum()

    # Required for the query. Message from fef25-75 is computed manually and won't use this precomp message
    uniform_from_fef2575 = {
        "['ecFEF25-75 % ecFEV1 (%)', 'Airway resistance (%)'] -> Airway resistance (%)": arr
    }
    m_from_hfev1_key = "Healthy FEV1 (L) -> ['Underlying FEV1 (L)', 'Healthy FEV1 (L)', 'Airway resistance (%)']"
    m_from_hfev1_dict = {}
    m_from_fev_factor_key = "['Underlying FEV1 (L)', 'Healthy FEV1 (L)', 'Airway resistance (%)'] -> Airway resistance (%)"
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
        for h, h_s_obs_state in enumerate(h_s_obs_states):
            HFEV1_bin_idx, S_obs_idx = h_s_obs_state

            vevidence_ar = (
                cutseth.build_vevidence_cutset_conditioned_ar_with_shape_factor(
                    AR,
                    h,
                    curr_date,
                    S_obs_idx,
                    prev_date,
                    next_date=None,
                    n_days_consec=n_days_consec,
                    debug=debug,
                )
            )
            res_dict["vevidence_ar"][n, :, h] = vevidence_ar.values

            if debug:
                print(
                    f"HFEV1_obs_idx: {HFEV1_bin_idx}, S_obs_idx: {S_obs_idx}, vevidence_ar: {vevidence_ar.values}"
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
                ) = cca.get_AR_and_p_log_D_given_M_obs_fev1_and_fef2575(
                    row,
                    inf_alg,
                    HFEV1,
                    HFEV1_bin_idx,
                    ecFEV1,
                    ecFEF2575prctecFEV1,
                    AR,
                    vevidence_ar,
                    {},
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
                    cca.get_AR_and_p_log_D_given_M_obs_fev1(
                        row,
                        inf_alg,
                        HFEV1,
                        HFEV1_bin_idx,
                        ecFEV1,
                        AR,
                        vevidence_ar,
                        uniform_from_fef2575,
                        m_from_hfev1_dict,
                        m_from_hfev1_key,
                        m_from_fev_factor_dict,
                        m_from_fev_factor_key,
                    )
                )
                dist_ecFEF2575prctecFEV1 = np.zeros(ecFEF2575prctecFEV1.card)
            else:
                raise ValueError("Both ecFEV1 and ecFEF25-75 must be observed")

            log_p_D_given_M[n, h] = log_p_D_given_M_for_row
            # If dist_AR contains at least one nan then print
            if np.isnan(dist_AR).any():
                print(f"Warning - dist_AR contains nan for ID {id}, row {n}, h {h}")
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
    return log_p_D_given_M, res_dict, AR_given_M_and_past_D, AR_given_M_and_same_day_D


def run_backward_sweep_to_get_ar_posteriors(
    df,
    h_s_obs_states,
    AR,
    AR_given_M_and_past_D,
    AR_given_M_and_same_day_D,
    debug,
):

    N = len(df)
    H = len(h_s_obs_states)

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
        if de > 1:
            ValueError(f"Days elapsed is {de}, should be at most 1")

        for h, h_s_obs_state in enumerate(h_s_obs_states):
            _, S_obs_idx = h_s_obs_state

            next_AR = AR_given_M_and_future_D[n + 1, :, h]
            curr_AR = AR_given_M_and_future_D[n, :, h]
            past_AR = AR_given_M_and_past_D[n, :, h]

            # Compute contribution from future days' data
            next_AR_m = np.matmul(next_AR, AR.change_cpt[:, :, S_obs_idx])
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
    HFEV1, h_s_obs_states, log_p_D_given_M, AR_given_M_and_all_D
):

    H = len(h_s_obs_states)

    # For each HFEV1 model, given HFEV1_obs_idx_list, we compute the log probability of the model given the data
    # log(P(M|D)) = 1/N * sum_n log(P(D|M)) + Cn_avg + log(P(M))
    log_p_M_given_D = np.zeros(H)
    for h, h_s_obs_state in enumerate(h_s_obs_states):
        HFEV1_bin_idx, S_obs_idx = h_s_obs_state
        log_p_M_hfev1 = np.log(HFEV1.cpt[HFEV1_bin_idx])
        # Do nan sums to remove contributions from days without data
        log_p_M_given_D[h] = np.nansum(log_p_D_given_M[:, h]) + log_p_M_hfev1

    # Exponentiating very negative numbers gives too small numbers
    # Setting the highest number to 1 to prevent numerical issues
    log_p_M_given_D -= log_p_M_given_D.max()
    p_M_given_D = np.exp(log_p_M_given_D)
    p_M_given_D /= p_M_given_D.sum()
    AR_given_M_and_D = np.matmul(AR_given_M_and_all_D, p_M_given_D)
    return p_M_given_D, AR_given_M_and_D, AR_given_M_and_all_D


def fuse_results_to_compute_P_S_given_D(
    id,
    HFEV1,
    S,
    log_p_D_given_M,
):
    # P(D|S) = sum_h ( P(D|S, H) P(H) )
    # P(D|S, H) = P(D|M)

    # Multiply the data over all days
    log_p_D_given_M = log_p_D_given_M.sum(axis=0)

    # Exit log space without having 0s
    max = log_p_D_given_M.max()
    log_p_D_given_M -= max
    p_D_given_M = np.exp(log_p_D_given_M)

    p_D_given_M = p_D_given_M.reshape((-1, S.card))
    # Handle case where the smallest possible inferable HFEV1 is larger than 1.
    argmin_hfev1 = HFEV1.card - p_D_given_M.shape[0]
    hfev1_cpt = HFEV1.cpt[argmin_hfev1:]
    p_D_given_S = np.matmul(hfev1_cpt, p_D_given_M)

    # Check for 0 probabilities
    if np.any(p_D_given_S == 0):
        print(
            f"Warning - ID {id}, P(S|M) has 0 probabilities p_D_given_S: {p_D_given_S}"
        )

    # Back to log space
    log_p_D_given_S = np.log(p_D_given_S)
    # Add back max to compare to merge result with other individuals'
    log_p_D_given_S += max

    return log_p_D_given_S
