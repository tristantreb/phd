import numpy as np


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

    return np.polyval(params, AR.midbins)
