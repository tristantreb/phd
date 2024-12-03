import concurrent.futures
import itertools

import src.data.breathe_data as bd
import src.data.helpers as dh
import src.inf_cutset_conditioning.cutset_cond_algs as cca

df = bd.load_meas_from_excel("BR_O2_FEV1_FEF2575_conservative_smoothing_with_idx")


def process_id(inf_settings):

    ar_change_cpt_suffix, ar_prior, id = inf_settings

    dftmp, start_idx, end_idx = dh.find_longest_consec_series(df[df.ID == id])

    print(
        f"Processing {inf_settings}, with {len(dftmp)} entries (start_index, end_index): ({start_idx}, {end_idx})"
    )

    # ecfev1_noise_model_suffix = "_std_0.068"
    ecfev1_noise_model_suffix = "_std_0.23"
    # ecfev1_noise_model_suffix = "_std_add_mult"

    (
        fig,
        p_M_given_D,
        AR_given_M_and_D,
    ) = cca.run_long_noise_model_through_time(
        # ) = cca.run_long_noise_model_through_time_light(
        dftmp,
        ar_prior=ar_prior,
        ar_change_cpt_suffix=ar_change_cpt_suffix,
        ecfev1_noise_model_suffix=ecfev1_noise_model_suffix,
        fef2575_cpt_suffix="",
        debug=False, 
        save=True,
    )
    return -1


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
        # "405",
        # "272",
        # "201",
        # "203",
        # "527",
    ]

    ar_priors = [
        # "uniform",
        # "uniform message to HFEV1",
        "breathe (2 days model, ecFEV1, ecFEF25-75)",
        # "breathe (2 days model, ecFEV1 addmultnoise, ecFEF25-75)",
    ]

    ar_change_cpt_suffix = [
        # "_shift_span_[-20;20]_joint_sampling_3_days_model", 
        # "_shift_span_[-20;20]_joint_sampling_3_days_model_ecfev1std0.23",
        # "_shift_span_[-20;20]_joint_sampling_3_days_model_ecfev1addmultnoise"
        "_shift_span_[-20;20]_joint_sampling_3_days_model_ecfev1std0.068"
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
