import numpy as np
import scipy.integrate as integrate
from scipy.stats import laplace, norm

import models.helpers as mh


def PDF_conv_uni_laplace_additive(z, y1, y2, s, abserr_tol=1e-10):
    """
    PDF of a convolution of a uniform and a gaussian distribution

    Return P(z | y1 < y < y2)
    """
    A = 1 / (y2 - y1)
    B = 1 / (2 * s)

    def conv_fn(y, z, s):
        return 1 / y * np.exp(np.abs(z - y) / s)

    val, abserr = integrate.quad(conv_fn, y1, y2, args=(z, s))
    if abserr > abserr_tol:
        raise ValueError(
            f"Absolute error after solving the integral is too high {abserr}"
        )

    return A * B * val


def pdf_laplace(x, mu, s):
    return 1 / (2 * s) * np.exp(-np.abs(x - mu) / s)


def p_uniform_x_laplace(z1, z2, y1, y2, s, abserr_tol=1e-10, debug=True):
    """
    Y ~ U(y1, y2)
    Z ~ Laplace(mu=Y, s=s)
    Returns P(z1 < z < z2 | y1 < y < y2)

    Validated against sampling
    """

    def conv_fn(z, y, s):
        return pdf_laplace(z, y, s) / (y2 - y1)

    val, abserr = integrate.dblquad(
        conv_fn, y1, y2, z1, z2, args=[s], epsabs=abserr_tol
    )
    if abserr > abserr_tol and debug:
        print(
            f"Warning - Absolute error after solving the integral is too high {abserr}, z1 = {z1}, z2 = {z2}, y1 = {y1}, y2 = {y2}"
        )

    return val


def p_uniform_x_gaussian(z1, z2, y1, y2, s, abserr_tol=1e-10, debug=True):
    """
    Y ~ U(y1, y2)
    Z ~ N(mu=Y, s=s)
    Returns P(z1 < z < z2 | y1 < y < y2)
    """

    def conv_fn(z, y, s):
        return norm.pdf(z, y, s) / (y2 - y1)

    val, abserr = integrate.dblquad(
        conv_fn, y1, y2, z1, z2, args=[s], epsabs=abserr_tol
    )
    if abserr > abserr_tol and debug:
        print(
            f"Warning - Absolute error after solving the integral is too high {abserr}, z1 = {z1}, z2 = {z2}, y1 = {y1}, y2 = {y2}"
        )

    return val


def p_uniform_x_gmm(
    z1, z2, y1, y2, s1, s2, w1, laplace_main, abserr_tol=1e-8, debug=True
):
    """
    Y ~ U(y1, y2)
    Z ~ w1 * N(mu=Y, s=s1) + w2 * N(mu=Y, s=s2)
    Returns P(z1 < z < z2 | y1 < y < y2)
    """
    w2 = 1 - w1

    def conv_fn(z, y, s1, s2, w1, w2, laplace_main=False):
        if laplace_main:
            return (w1 * laplace.pdf(z, y, s1) + w2 * norm.pdf(z, y, s2)) / (y2 - y1)
        else:
            return (w1 * norm.pdf(z, y, s1) + w2 * norm.pdf(z, y, s2)) / (y2 - y1)

    val, abserr = integrate.dblquad(
        conv_fn,
        y1,
        y2,
        lambda y: z1,
        lambda y: z2,
        args=[s1, s2, w1, w2, laplace_main],
        epsabs=abserr_tol,
    )
    if abserr > abserr_tol and debug:
        print(
            f"Warning - Absolute error after solving the integral is too high {abserr}, z1 = {z1}, z2 = {z2}, y1 = {y1}, y2 = {y2}"
        )

    return val


def get_breathe_prior_from_1_day_model_o2sat_ecFEV1():
    prior = np.array(
        [
            3.30017877e-02,
            2.94332809e-02,
            2.84614482e-02,
            2.81412788e-02,
            2.80715428e-02,
            2.82201463e-02,
            2.85892890e-02,
            2.91370387e-02,
            2.98440633e-02,
            3.06870369e-02,
            3.16515009e-02,
            3.27659391e-02,
            3.40436812e-02,
            3.54580906e-02,
            3.69616283e-02,
            3.84909562e-02,
            3.99707817e-02,
            4.01369105e-02,
            3.77586072e-02,
            3.43664921e-02,
            3.14186539e-02,
            2.94729070e-02,
            2.79070486e-02,
            2.67226722e-02,
            2.59757994e-02,
            2.56031345e-02,
            2.53547631e-02,
            2.48108307e-02,
            2.35277866e-02,
            2.19954118e-02,
            1.98629847e-02,
            1.72215473e-02,
            1.40814049e-02,
            1.14758315e-02,
            8.35226825e-03,
            5.45547550e-03,
            2.82376684e-03,
            1.15447994e-03,
            4.89654330e-04,
            3.83187888e-04,
            3.77857055e-04,
            2.55912690e-04,
            7.89846586e-05,
            6.09149399e-06,
            4.48797604e-08,
        ]
    )
    prior = prior / np.sum(prior)
    return prior


def get_breathe_prior_from_2_days_model_ecFEV1_ecFEF2575():
    prior = np.array(
        [
            6.11640552e-03,
            1.06277102e-02,
            1.67953608e-02,
            2.40280203e-02,
            2.81953570e-02,
            3.08307897e-02,
            3.31139141e-02,
            3.49966946e-02,
            3.56556967e-02,
            3.49566471e-02,
            3.31038263e-02,
            3.20199153e-02,
            3.21722512e-02,
            3.31384393e-02,
            3.44443241e-02,
            3.59152936e-02,
            3.74083697e-02,
            3.83917327e-02,
            3.78846072e-02,
            3.70470948e-02,
            3.65646442e-02,
            3.49691951e-02,
            3.36754074e-02,
            3.24133419e-02,
            3.09303100e-02,
            2.95958755e-02,
            2.84330163e-02,
            2.71876474e-02,
            2.55290796e-02,
            2.33528993e-02,
            2.07703567e-02,
            1.79969619e-02,
            1.49841139e-02,
            1.20675198e-02,
            9.27738935e-03,
            6.78022379e-03,
            4.42289991e-03,
            2.12593694e-03,
            6.34910446e-04,
            3.17294432e-04,
            4.22426531e-04,
            4.27418502e-04,
            2.31645130e-04,
            4.47277976e-05,
            2.30775204e-06,
        ]
    )
    prior = prior / np.sum(prior)
    return prior


def get_breathe_prior_from_2_days_model_ecFEV1_ecFEF2575_ecfev1addmultnoise():
    return np.array(
        [
            4.90743057e-03,
            8.36492902e-03,
            1.26815952e-02,
            1.83332552e-02,
            2.40080204e-02,
            2.85445420e-02,
            3.12612484e-02,
            3.27961674e-02,
            3.37290575e-02,
            3.39439581e-02,
            3.32093967e-02,
            3.24260714e-02,
            3.23609904e-02,
            3.29212825e-02,
            3.38210869e-02,
            3.49095875e-02,
            3.62891533e-02,
            3.75670010e-02,
            3.80198133e-02,
            3.73843556e-02,
            3.64256030e-02,
            3.51009030e-02,
            3.37524996e-02,
            3.23147030e-02,
            3.12438767e-02,
            3.03535633e-02,
            2.95491702e-02,
            2.88109431e-02,
            2.77691472e-02,
            2.58842258e-02,
            2.35544450e-02,
            2.07081298e-02,
            1.77948329e-02,
            1.48048263e-02,
            1.17147514e-02,
            8.76708676e-03,
            6.08214144e-03,
            3.72247805e-03,
            1.88729569e-03,
            8.40713780e-04,
            4.96209733e-04,
            4.43626034e-04,
            3.37391437e-04,
            1.39969284e-04,
            2.25252982e-05,
        ]
    )


def get_uniform_prior_in_log_space(AR):
    """
    Returns a uniform prior in log space for the AR variable.
    """
    prior = np.log(AR.midbins)
    prior = prior / np.sum(prior)
    return prior


def get_prior_for_uniform_hfev1_message(AR):
    """
    Returns the AR prior that, when multiplied by a point mass message from ecFEV1, outputs a uniform message going to HFEV1.

    To compute the prior,
    1/ we take the CPT of dim card(ecFEV1) x card(HFEV1) x card(AR),
    2/ we select one bin of ecFEV1 to simulate a point mass message, thus reducing the CPT's dimension by one,
    3/ we create HFEV1_uni a uniform message for HFEV1
    4/ we compute ar_output = HFEV1_uni * CPT
    5/ this ar_output is the prior of AR that gives a uniform message to HFEV1 when multiplied by the point mass message from ecFEV1.
    6/ we then fitted this ar_output with a polynomial function
    7/ saving the parameters allows to recreate the message with the right shape with any resolution for the AR variable
    """
    # We fitted a polynomial of degree 12
    params = np.array(
        [
            6.84808411e-23,
            -3.01603307e-20,
            5.86472319e-18,
            -6.60296098e-16,
            4.75372724e-14,
            -2.28184316e-12,
            7.39699647e-11,
            -1.60333825e-09,
            2.27139833e-08,
            -1.85804099e-07,
            1.94333647e-06,
            9.79127935e-05,
            1.00017236e-02,
        ]
    )

    ar_prior = np.polyval(params, AR.midbins)
    ar_prior = ar_prior / np.sum(ar_prior)
    return ar_prior


def calc_cpt(
    AR_next_day: mh.VariableNode,
    AR_curr_day: mh.VariableNode,
    DE: mh.DiscreteVariableNode,
    shift_p,
    shift_val,
    tol=1e-6,
    debug=False,
):
    cpt = np.zeros([AR_next_day.card, AR_curr_day.card, DE.card])

    for de in range(DE.card):
        # For each shift value, get the mapping AR -> AR_next_day for each shifted bin in AR
        # Weight the result by the probability of that shift
        # Add it to the CPT for this day
        for s in range(len(shift_p)):
            cpt[:, :, de] += shift_p[de, s] * calc_cpt_X_plus_k(
                AR_curr_day,
                AR_next_day,
                shift_val[s],
                tol=tol,
                debug=debug,
            )
        # Normalise the CPT for this amount of days elapsed
        total = np.sum(cpt[:, :, de])
        cpt[:, :, de] /= total

    # Check that the sum of probabilities is 1
    total = np.sum(cpt)
    assert (
        abs(total - 1) < tol
    ), f"The sum of the probabilities should be 1, got sum(cpt)={total}])"
    return cpt


def calc_cpt_X_plus_k(
    Z: mh.VariableNode,
    X: mh.VariableNode,
    k,
    tol=1e-6,
    debug=False,
):
    """
    Computes the CPT for P(Z|X, Y), when Z is shifted from X by a constant value k
    Z = X + k
    X: parent variable
    Z: child variable
    k: constant, positive or negative

    We compute the CPT with a shift and conquer method:
    1) Start with a CPT zeroed out probabilities
    2) Shift all X bin intervals by the drop amount
    3) For each shifted X bin, spread the X bin evenly onto the overlapping Z bins
    4) Normalise the CPT

    This allows the function to be agnostic of how X and Z are binned.

    - What happens when the function is shifted outside the boundary? -> Raise an error as it shouldn't happen by how the model is built
    """
    nbinsX = X.card
    nbinsZ = Z.card

    cpt = np.zeros([nbinsZ, nbinsX])

    for i in range(nbinsX):
        shifted_X_bin_low = X.bins[i] + k
        shifted_X_bin_up = (X.bins[i] + X.bin_width) + k
        if debug:
            print(
                f"Shifting X bin {i} from [{X.bins[i]};{X.bins[i]+X.bin_width}] to [{shifted_X_bin_low};{shifted_X_bin_up}], shift amount={k}%"
            )
        # If the shifted bin is outside the boundaries of Z, continue:
        if shifted_X_bin_low >= Z.bins[-1] or shifted_X_bin_up <= Z.bins[0]:
            if debug:
                print("Shift outside boundaries")
            continue
        # Handle the case where the shifted bin is partially outside the boundaries
        # Adjust the boundaries of the shifted bin to be within the boundaries of Z
        if shifted_X_bin_low < Z.bins[0]:
            if debug:
                print("Shift partially outside boundaries, adjusting lower boundary")
            shifted_X_bin_low = Z.bins[0]
        if shifted_X_bin_up > Z.bins[-1]:
            if debug:
                print("Shift partially outside boundaries, adjusting upper boundary")
            shifted_X_bin_up = Z.bins[-1]

        bin_contribution = mh.get_bin_contribution_to_cpt(
            [shifted_X_bin_low, shifted_X_bin_up], Z.bins, debug=debug
        )
        # if debug:
        #     print(f"i={i}/{nbinsX-1}, j={j}/{nbinsY-1}, z={bin_contribution}")
        cpt[:, i] += bin_contribution

    # Normalise all cpt(:, i, j) to 1
    total = np.sum(cpt[:, i])
    if debug:
        print(f"Results before normalisation sum(cpt[:, {i}])={total}")
    cpt[:, i] /= total

    # Raise if sum of probabilities is larger than 1
    total = np.sum(cpt[:, i])
    assert (
        abs(total - 1) < tol
    ), f"The sum of the probabilities should be 1, got sum(cpt[:, {i}])={total}])"

    return cpt
