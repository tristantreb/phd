import concurrent.futures
import itertools

import src.data.breathe_data as bd
import src.inf_cutset_conditioning.cutset_cond_algs as cca

df = bd.load_meas_from_excel("BR_O2_FEV1_FEF2575_conservative_smoothing_with_idx")


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
    elif id == "527":
        dftmp = df[df["ID"] == "527"]

    (
        fig,
        p_M_given_D,
        AR_given_M_and_D,
    ) = cca.run_long_noise_model_through_time(
    # ) = cca.run_long_noise_model_through_time_light(
        dftmp,
        ar_prior=ar_prior,
        ar_change_cpt_suffix=ar_change_cpt_suffix,
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
        # "101",
        # # Also from consec values
        # "405",
        "272",
        # "201",
        # "203",
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
