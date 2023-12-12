import src.models.helpers as mh


def add_gaussian_noise(o2sat, bin_width, absolute_max=100):
    """
    Add gaussian noise to O2 saturation
    """
    max_deviation = 2
    std = 1
    o2sat_with_noise = mh.variableNode(
        "O2 saturation (%)",
        o2sat - max_deviation,
        min(o2sat + max_deviation, absolute_max),
        bin_width,
        prior={"type": "gaussian", "mu": o2sat, "sigma": std},
    )
    return o2sat_with_noise
