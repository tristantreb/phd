import concurrent.futures
import itertools

import numpy as np

import src.data.breathe_data as bd
import src.data.helpers as dh
import src.inf_cutset_conditioning.cutset_cond_algs_learn_ar_change as cca_ar_change

# import src.inf_cutset_conditioning.cutset_cond_algs as cca

df = bd.load_meas_from_excel("BR_O2_FEV1_FEF2575_conservative_smoothing_with_idx")


def process_id(inf_settings):

    ar_change_cpt_suffix, ar_prior, id = inf_settings
    n_days_consec = 2
    ecfev1_noise_model_suffix = "_std_0.068"
    # ecfev1_noise_model_suffix = "_std_0.23"
    # ecfev1_noise_model_suffix = "_std_add_mult"

    dftmp, start_idx, end_idx = dh.find_longest_consec_series(
        df[df.ID == id], n_days=n_days_consec
    )

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

    (
        # p_M_given_D,
        # log_p_D_given_M,
        # AR_given_M_and_D,
        # AR_given_M_and_all_D,
        log_p_S_given_D,
        res_dict,
    ) = cca_ar_change.run_long_noise_model_through_time(
        # ) = cca.run_long_noise_model_through_time_light(
        dftmp,
        ar_prior=ar_prior,
        ar_change_cpt_suffix=ar_change_cpt_suffix,
        ecfev1_noise_model_suffix=ecfev1_noise_model_suffix,
        fef2575_cpt_suffix="",
        debug=False,
        save=True,
    )

    max = log_p_S_given_D.max()

    print(f"id {id}, log_p_S_given_D: {log_p_S_given_D - max}")

    return {id: log_p_S_given_D}


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
        "527",
    ]

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
        "_shape_factor",
    ]

    # Zip the three elements together, to create a list of tuples of size card x card x card
    inf_settings = list(
        itertools.product(ar_change_cpt_suffix, ar_priors, interesting_ids)
    )

    # num_cores = os.cpu_count()
    # with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
    with concurrent.futures.ProcessPoolExecutor() as executor:
        log_p_S_given_D_list = list(executor.map(process_id, inf_settings))

    # Join the list of dictionaries into a single dictionary
    log_p_S_given_D = {k: v for d in log_p_S_given_D_list for k, v in d.items()}
    print(log_p_S_given_D)


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
