import numpy as np
from scipy.stats import norm

import src.models.helpers as mh


def get_std_func(ar):
    """
    Std of IA as a function of AR
    std = f(ar)
    """
    return 0.000085 * ar**2 + 0.000000018 * ar**4 + 0.485


def get_IA_proba(bins, std):
    """
    Returns a probability for each possible bin in IA

    The percentage of inactive alveoli is modelled as the positive side of the gaussian distribution N(0, std)
    """
    # Raise if bins contain negative values
    assert np.all(bins >= 0), f"bins should be positive, got bins={bins}"

    # Since bins has only positive values, applying the pdf will only return the positive side of the gaussian
    return norm.pdf(bins, loc=0, scale=std)


def calc_cpt(
    IA: mh.variableNode,
    AR: mh.variableNode,
):
    """
    Computes the CPT for P(IA|AR)
    IA: inactive alveoli
    AR: airway resistance
    """

    nbinsIA = len(IA.bins)
    nbinsAR = len(AR.bins)

    cpt = np.zeros((nbinsIA, nbinsAR))

    # Create a for loop over the values of AR.bins
    for i in range(nbinsAR):
        # Get the std for the current value of AR
        std = get_std_func(AR.bins_arr[i])
        # Get the IA for the current std
        cpt[:, i] = get_IA_proba(IA.bins_arr, std)

        # Raise if sum of probabilities is larger than 1
        total = np.sum(cpt[:, i])
        assert (
            abs(total - 1) < IA.tol
        ), f"The sum of the probabilities should be 1, got sum(cpt[:, {i}])={total}])"

    return cpt
