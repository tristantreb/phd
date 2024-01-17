import numpy as np

import src.models.helpers as mh


def generate_o2sat_measurement(UO2Sat: mh.variableNode, std_gauss):
    real = np.random.uniform(UO2Sat.a, UO2Sat.b)

    # Add gaussian noise to the number with a standard deviation of std_gauss
    noisy_real = np.random.normal(real, std_gauss)

    # Round x to the nearest 1
    rounded = round(noisy_real)
    return real, noisy_real, rounded


def generate_underlying_uo2sat_distribution(
    UO2Sat: mh.variableNode, o2sat_obs, repetitions, std_gauss, show_std=False
):
    uo2sat_arr = np.array([])
    for _ in range(repetitions):
        real, _, rounded = generate_o2sat_measurement(UO2Sat, std_gauss)
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


def calc_cpt(O2Sat: mh.variableNode, UO2Sat: mh.variableNode):
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

    cpt = np.zeros((len(O2Sat.bins), len(UO2Sat.bins)))

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
