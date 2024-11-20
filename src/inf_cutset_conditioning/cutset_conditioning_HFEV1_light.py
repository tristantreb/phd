import concurrent.futures

import src.data.breathe_data as bd
import src.inf_cutset_conditioning.cutset_conditioning_algs as cca

df = bd.load_meas_from_excel("BR_O2_FEV1_FEF2575_conservative_smoothing_with_idx")


def process_id(id):
    ar_prior = "breathe (2 days model, ecFEV1, ecFEF25-75)"
    ar_prior = "uniform"

    dftmp = df[df.ID == id]
    return cca.compute_log_p_D_given_M_for_HFEV1_light(
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
