import numpy as np
import scipy.integrate as integrate


def PDF_conv_uni_gausian_additive(z, y1, y2, sigma, abserr_tol=1e-10):
    """
    PDF of a convolution of a uniform and a gaussian distribution

    Return P(z | y1 < y < y2)
    """
    A = 1 / (y2 - y1)
    B = 1 / (sigma * np.sqrt(2 * np.pi))

    def conv_fn(y, z, sigma):
        return 1 / y * np.exp(-1 / 2 * (z - y) ** 2 / sigma**2)

    val, abserr = integrate.quad(conv_fn, y1, y2, args=(z, sigma))
    if abserr > abserr_tol:
        raise ValueError(
            f"Absolute error after solving the integral is too high {abserr}"
        )

    return A * B * val


def PDF_conv_uni_gausian_multiplicative(z, y1, y2, abserr_tol=1e-10):
    """
    A more correct approach is a linear noise: both additive and multiplicative noise are present
    PDF of a convolution of a uniform and a gaussian distribution

    Return P(z | y1 < y < y2)
    """
    A = 1 / (y2 - y1)

    def sigma_fn(fev1):
        if fev1 <= 2:
            return 0.03440651
        elif fev1 > 2 and fev1 < 3:
            return 0.04515183
        else:
            return 0.05417496

    def conv_fn(y, z):
        return np.exp(-1 / 2 * (z - y) ** 2 / sigma_fn(z) ** 2) / (
            sigma_fn(z) * np.sqrt(2 * np.pi) * y
        )

    val, abserr = integrate.quad(conv_fn, y1, y2, args=(z))
    if abserr > abserr_tol:
        raise ValueError(
            f"Absolute error after solving the integral is too high {abserr}: y1={y1}, y2={y2}, z={z}, sigma={sigma_fn(z)}"
        )

    return A * val
