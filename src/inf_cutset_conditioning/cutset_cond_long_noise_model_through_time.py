import concurrent.futures
import itertools

# import inf_cutset_conditioning.cutset_cond_algs as cca
import numpy as np
import pandas as pd

import data.breathe_data as bd
import data.helpers as dh
import inf_cutset_conditioning.cutset_cond_algs_learn_ar_change as cca_ar_change
import inf_cutset_conditioning.cutset_cond_algs_learn_ar_change_noo2sat as cca_ar_change_noo2sat

# df = bd.load_meas_from_excel("BR_O2_FEV1_FEF2575_conservative_smoothing_with_idx_granular")
df = bd.load_meas_from_excel("BR_O2_FEV1_FEF2575_conservative_smoothing_with_idx")
# With step change

# df_step_change = df.loc[2445:2475]
# # Create a new date range with consecutive dates
# start_date = df_step_change["Date Recorded"].min()
# end_date = start_date + pd.Timedelta(days=len(df_step_change) - 1)
# date_range = pd.date_range(start=start_date, end=end_date, freq="D")
# df_step_change["Date Recorded"] = date_range
# # df_step_change = df_step_change[df_step_change['ecFEV1'].diff() <= 0][1:-1]

# df_constant = df.loc[0:30]

# ids = ["104", "101"]
# df = pd.concat([df_step_change, df_constant])


def process_id(inf_settings):

    get_p_s_given_d = True

    ar_change_cpt_suffix, ar_prior, id = inf_settings
    n_days_consec = 3
    # ecfev1_noise_model_suffix = "_std_0.068"
    # ecfev1_noise_model_suffix = "_std_0.23"
    ecfev1_noise_model_suffix = "_std_add_mult_ecfev1"
    # ecfev1_noise_model_suffix = "_std_add_mult_fev1_midbin"

    dftmp, start_idx, end_idx = dh.find_longest_consec_series(
        df[df.ID == id], n_days=n_days_consec
    )
    # Select only 30 days per ID
    dftmp = dftmp.head(30)

    print(
        f"Processing {inf_settings}, with {len(dftmp)} entries (start_index, end_index): ({start_idx}, {end_idx})"
    )

    ecfef2575_cols = [
        "ecFEF2575%ecFEV1",
        "idx ecFEF2575%ecFEV1",
        "idx ecFEF25-75 % ecFEV1 (%)",
    ]
    ecfev1_cols = [
        "ecFEV1",
        "idx ecFEV1 (L)",
    ]
    # Obs FEV1 and FEF25-75
    #
    # Obs FEV1
    # dftmp[ecfef2575_cols] = np.nan
    # Obs no data
    # dftmp[ecfev1_cols + ecfef2575_cols] = np.nan

    out = cca_ar_change_noo2sat.run_long_noise_model_through_time(
        # ) = cca_ar_change.run_long_noise_model_through_time(
        # ) = cca.run_long_noise_model_through_time_light(
        dftmp,
        ar_prior=ar_prior,
        ar_change_cpt_suffix=ar_change_cpt_suffix,
        ecfev1_noise_model_suffix=ecfev1_noise_model_suffix,
        fef2575_cpt_suffix="",
        n_days_consec=n_days_consec,
        light=False,
        debug=False,
        get_p_s_given_d=get_p_s_given_d,
        save=True,
    )

    if get_p_s_given_d:
        (
            log_p_S_given_D,
            res_dict,
        ) = out
        res = {id: log_p_S_given_D}
        # Write results to file p_s_given_d.json
        with open(
            f"{dh.get_path_to_src()}/inf_cutset_conditioning/p_s_given_d.json", "a"
        ) as f:
            f.write(str(res) + "\n")
        f.close()
    else:
        (
            fig,
            p_M_given_D,
            p_HFEV1_given_D,
            log_p_D_given_M,
            AR_given_M_and_D,
            AR_given_M_and_all_D,
            res_dict,
        ) = out

    return -1


# Run the function in parallel using ProcessPoolExecutor
if __name__ == "__main__":

    interesting_ids = [
        "132",
        "146",
        "177",
        "180",
        "202",
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
        "527",
        # For step change
        "104",
    ]
    # interesting_ids = df.ID.unique()
    # interesting_ids = ids

    ar_priors = [
        # "uniform",
        # "uniform message to HFEV1",
        # "breathe (2 days model, ecFEV1, ecFEF25-75)",
        "breathe (2 days model, ecFEV1 addmultnoise, ecFEF25-75)",
    ]

    ar_change_cpt_suffix = [
        # "_shift_span_[-20;20]_joint_sampling_3_days_model",
        # "_shift_span_[-20;20]_joint_sampling_3_days_model_ecfev1std0.23",
        # "_shift_span_[-20;20]_joint_sampling_3_days_model_ecfev1addmultnoise",
        # "_shift_span_[-20;20]_joint_sampling_3_days_model_ecfev1std0.068",
        # "_shape_factor_Gmain0.2_Gtails10_w0.73",
        # "_shape_factor_grid_search_2",
        # "_shape_factor_weight_card11",
        # "_shape_factor_main_tail_card28",
        # "_shape_factor_main_tail_card23",
        # "_shape_factor_single_laplace_card9",
        # "_shape_factor_single_laplace_0.5",
        # "_shape_factor_single_laplace_1.5",
        # "_shape_factor_single_laplace_card10",
        "_shape_factor_single_laplace_card3",
    ]

    # Zip the three elements together, to create a list of tuples of size card x card x card
    inf_settings = list(
        itertools.product(ar_change_cpt_suffix, ar_priors, interesting_ids)
    )

    # num_cores = os.cpu_count()
    # with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
    with concurrent.futures.ProcessPoolExecutor() as executor:
        list(executor.map(process_id, inf_settings))


# def main():
#     process_id(("uniform", "101"))
#     return -1

# main()

# import cProfile
# import pstats

# prof = cProfile.Profile()
# prof.run("main()")
# prof.dump_stats("cuset_cond_profiling.prof")

# stream = open("cuset_cond_profiling.txt", "w")
# stats = pstats.Stats("cuset_cond_profiling.prof", stream=stream)
# stats.sort_stats("cumtime")
# stats.print_stats()
