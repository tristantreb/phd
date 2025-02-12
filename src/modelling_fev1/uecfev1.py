import numpy as np
import scipy.integrate as integrate

# def PDF_conv_uni_gausian_additive(z, y1, y2, sigma, abserr_tol=1e-10):
#     """
#     PDF of a convolution of a uniform and a gaussian distribution

#     Return P(z | y1 < y < y2)
#     """
#     A = 1 / (y2 - y1)
#     B = 1 / (sigma * np.sqrt(2 * np.pi))

#     def conv_fn(y, z, sigma):
#         # return 1 / y * np.exp(-1 / 2 * (z - y) ** 2 / sigma**2)
#         return 1 / y * np.exp(-1 / 2 * (z - y) ** 2 / sigma**2)

#     val, abserr = integrate.quad(conv_fn, y1, y2, args=(z, sigma))
#     if abserr > abserr_tol:
#         raise ValueError(
#             f"Absolute error after solving the integral is too high {abserr}"
#         )

#     return A * B * val


def sigma_fn(fev1, use_ecfev1=True):
    """
    std_ecfev1(ecfev1) = ax + b
    # ecfev1 has less noise than fev1 because of the filtering
    a = 0.00510174, multiplicative noise
    b = 0.03032977, additive noise

    std_fev1(fev1) = ax + b
    a = 0.00527939, multiplicative noise
    b = 0.03396603, additive noise
    """
    if use_ecfev1:
        return 0.00510174 * fev1 + 0.03032977
    else:
        return 0.00527939 * fev1 + 0.03396603


def p_uniform_x_gauss_add_mult_noise(z1, z2, y1, y2, abserr_tol=1e-10):
    """
    A more correct approach is a linear noise: both additive and multiplicative noise are present
    PDF of a convolution of a uniform and a gaussian distribution

    Return P(z1 < z < z2 | y1 < y < y2)
    """

    def pdf_gauss(y, z):
        """
        y: is the mean
        z: is the value
        """
        return np.exp(-((z - y) ** 2) / (2 * sigma_fn(z) ** 2)) / (
            sigma_fn(z) * np.sqrt(2 * np.pi)
        )

    def conv_fn(y, z):
        """
        Mean is uniformly distributed between y1 and y2
        """
        return pdf_gauss(y, z) / (y2 - y1) / y

    val, abserr = integrate.dblquad(conv_fn, z1, z2, y1, y2, epsabs=abserr_tol)
    if abserr > abserr_tol:
        raise ValueError(
            f"Absolute error after solving the integral is too high {abserr}: y1={y1}, y2={y2}, z1={z1}, z2={z2} sigma=[{sigma_fn(z1)}, {sigma_fn(z2)}]"
        )

    return val


def PDF_conv_uni_gausian_add_mult(z, y1, y2, abserr_tol=1e-10):
    """
    A more correct approach is a linear noise: both additive and multiplicative noise are present
    PDF of a convolution of a uniform and a gaussian distribution

    Return P(z | y1 < y < y2)
    """
    A = 1 / (y2 - y1)

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
