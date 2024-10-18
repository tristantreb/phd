import numpy as np
import scipy.integrate as integrate


def PDF_conv_uni_gausian(z, y1, y2, sigma, debug=False):
    """
    PDF of a convolution of a uniform and a gaussian distribution
    """
    A = 1 / (y2 - y1)
    B = 1 / (sigma * np.sqrt(2 * np.pi))

    def conv_fn(y, z, sigma):
        return 1 / y * np.exp(-1 / 2 * (z - y) ** 2 / sigma**2)

    val, abserr = integrate.quad(conv_fn, y1, y2, args=(z, sigma))
    if abserr > 1e-10:
        raise ValueError(
            f"Absolute error after solving the integral is too high {abserr}"
        )

    return A * B * val
