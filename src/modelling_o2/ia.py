import numpy as np
from scipy.stats import norm

import src.models.helpers as mh


def get_IA_breathe_prior():
    """
    Prior for IA learnt on the Breathe dataset

    See 2024-02-06_point_in_time_model_validation.ipynb
    Built for the following parametrisation IA = mh.VariableNode("Inactive alveoli (%)", 0, 30, 1, prior={"type": "uniform"})
    """
    prior = np.array(
        [
            5.36355675e-01,
            2.66767859e-01,
            1.13131207e-01,
            4.66606260e-02,
            1.98108665e-02,
            8.75698488e-03,
            4.09439933e-03,
            2.02727345e-03,
            1.01367484e-03,
            5.06547100e-04,
            2.76685880e-04,
            1.67053401e-04,
            1.10943203e-04,
            8.05764823e-05,
            5.91724343e-05,
            4.22907658e-05,
            3.61250560e-05,
            3.08420975e-05,
            1.75300246e-05,
            9.24248083e-06,
            1.30604702e-05,
            1.68367810e-05,
            1.08827456e-05,
            3.21698274e-06,
            4.05516399e-07,
            2.21446540e-08,
            4.76466662e-10,
            1.43163804e-12,
            0.00000000e00,
            0.00000000e00,
        ]
    )
    return prior.reshape(-1, 1)


def get_std_func(ar):
    """
    Std of IA as a function of AR
    std = f(ar)
    """
    return 0.000085 * ar**2 + 0.000000018 * ar**4 + 0.485


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


def calc_cpt(IA: mh.VariableNode, AR: mh.VariableNode, debug=True):
    """
    Computes the CPT for P(IA|AR)
    IA: inactive alveoli
    AR: airway resistance
    """

    nbinsIA = IA.card
    nbinsAR = AR.card

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
