import numpy as np

import src.models.helpers as mh


def generate_o2sat_measurement(a, b, std_gauss):
    real = np.random.uniform(a, b)

    # Add gaussian noise to the number with a standard deviation of std_gauss
    noisy_real = np.random.normal(real, std_gauss)

    # Round x to the nearest 1
    rounded = round(noisy_real)
    return real, noisy_real, rounded


def generate_o2sat_distribution(
    O2Sat_span, uo2sat_bin, repetitions, std_gauss, show_std=False
):
    """
    Generates the downwards distribution, P(O2Sat | UO2Sat = uo2sat_bin)
    O2Sat_span: upper and lower boundaries of the O2 saturation variable that constraint the sampled values
    uo2sat_bin: upper and lower boundary of the UO2Sat bin from which to sample from
    """
    o2sat_arr = np.array([])

    # This represents P(O2Sat | UO2Sat = [uo2sat_down, uo2sat_up[)
    for _ in range(repetitions):
        _, _, o2sat_obs = generate_o2sat_measurement(
            uo2sat_bin[0], uo2sat_bin[1], std_gauss
        )
        # Important: o2sat can't be above 100%.
        # Important: to avoid dealing with issues on the lower boundary we also exclude values below it
        ## That means the 3-4 first bins (50-54%) of O2Sat will have wrong values, but it's fine because o2 sat values
        ## below 70% are pratically impossible (smallest measurement in Breathe data is 84%)
        ## Adding a check to reflect this comment
        if O2Sat_span[0] > 60:
            raise ValueError(
                f"Can't have O2Sat's lower boundary above 60, got {O2Sat_span[0]}"
            )
        if o2sat_obs >= O2Sat_span[0] and o2sat_obs <= O2Sat_span[1]:
            o2sat_arr = np.append(o2sat_arr, o2sat_obs)

    if show_std:
        print(
            f"Std of o2sat values for UO2Sat bin {uo2sat_bin}: {round(o2sat_arr.std(), 4)}"
        )

    # o2sat_arr shall only contain integers
    hist, bin_edges = np.histogram(
        o2sat_arr, bins=np.arange(O2Sat_span[0], O2Sat_span[1] + 2, 1)
    )

    # Normalize the histogram if hist is not empty
    if sum(hist) > 0:
        hist = hist / sum(hist)

    return hist, bin_edges, o2sat_arr


def generate_underlying_uo2sat_distribution(
    UO2Sat: mh.VariableNode, o2sat_obs, repetitions, std_gauss, show_std=False
):
    """
    Generates the upwards distribution, P(UO2Sat | O2Sat)
    """
    uo2sat_arr = np.array([])
    for _ in range(repetitions):
        real, _, rounded = generate_o2sat_measurement(UO2Sat.a, UO2Sat.b, std_gauss)
        if rounded == o2sat_obs:
            uo2sat_arr = np.append(uo2sat_arr, real)

    if show_std:
        print(f"Std of values giving {o2sat_obs}: {round(uo2sat_arr.std(), 4)}")

    # Use np to histogram the values
    hist, bin_edges = np.histogram(
        uo2sat_arr,
        bins=np.arange(UO2Sat.a, UO2Sat.b + UO2Sat.bin_width, UO2Sat.bin_width),
    )

    # Normalize the histogram if hist is not empty
    if sum(hist) > 0:
        hist = hist / sum(hist)

    return hist, bin_edges, uo2sat_arr


def calc_cpt(O2Sat: mh.VariableNode, UO2Sat: mh.VariableNode):
    """
    The CPT is calculated using the generative o2 saturation noise model
    See 2024-01-08_o2sat_noise_model.ipynb

    O2Sat: O2 saturation
    UO2Sat: Unbiased O2 saturation. True O2 saturation (without technical/biological noise)
    """
    # Parameters
    ## Std of the gaussian noise
    std_gauss = 0.86
    ## Sampling size
    repetitions = 1000000

    cpt = np.zeros((O2Sat.card, UO2Sat.card))

    for i, o2sat_obs in enumerate(O2Sat.bin):
        # Generate underlying distribution
        uo2sat_hist, _, _ = generate_underlying_uo2sat_distribution(
            UO2Sat, o2sat_obs, repetitions, std_gauss
        )

        cpt[i, :] = uo2sat_hist

    # Normalise the cpt
    normaliser = cpt.sum(axis=0)
    for i, norm in enumerate(normaliser):
        if norm != 0:
            cpt[:, i] = cpt[:, i] / norm

        # Check that the sum of the column is 1
        assert np.isclose(
            cpt[:, i].sum(), 1, tol=UO2Sat.tol
        ), f"The sum of probabilities should be 1, got {cpt[:, i].sum()} while calculating P({O2Sat.name}|{UO2Sat.name}={UO2Sat.bins_str[i]})"

    return cpt
