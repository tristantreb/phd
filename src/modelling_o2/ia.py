import numpy as np
from scipy.stats import norm

import src.models.helpers as mh


def get_std_func(ar):
    """
    Std of IA as a function of AR
    std = f(ar)
    """
    return (0.000085 * ar**2 + 0.000000018 * ar**4 + 0.485) * 20


def get_IA_proba(bins, std, debug):
    """
    Returns a probability for each possible bin in IA

    The percentage of inactive alveoli is modelled as the positive side of the gaussian distribution N(0, std)
    """
    # Raise if bins contain negative values
    assert np.all(bins >= 0), f"bins should be positive, got bins={bins}"

    # Since bins has only positive values, applying the pdf will only return the positive side of the gaussian
    p = norm.pdf(bins, loc=0, scale=std)
    # Normalize
    p = p / np.sum(p)
    if debug:
        print(f"Proba from N(0, std), normalised: {p}")
    return p


def calc_cpt(IA: mh.variableNode, AR: mh.variableNode, debug=True):
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
        if debug:
            print(f"i={i}")
        # Get the std for the current value of AR
        std = get_std_func(AR.midbins[i])
        # Get the IA for the current std
        cpt[:, i] = get_IA_proba(IA.midbins, std, debug)

        # Raise if sum of probabilities is larger than 1
        total = np.sum(cpt[:, i])
        assert (
            abs(total - 1) < IA.tol
        ), f"The sum of the probabilities should be 1, got sum(cpt[:, {i}])={total}])"

    return cpt
