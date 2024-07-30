import numpy as np

import src.models.helpers as mh


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
    tol=1e-6,
    debug=False,
):

    return


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
    nbinsX = len(X.bins)
    nbinsZ = len(Z.bins)

    cpt = np.zeros([nbinsZ, nbinsX])
    if debug:
        print(f"Shape of cpt: {cpt.shape}")

        # # The drop amount for a Y bin is the average of the function over the bin
        # # Simplified to avg( f(bin_up) - f(bin_low) )
        # drop = np.mean([func(Y.bins[j] + Y.bin_width), func(Y.bins[j])])
        # if debug:
        #     print(f"Drop for Y bin {j} ([{Y.bins[j]};{Y.bins[j]+Y.bin_width}]): {drop}")

        # For computational efficiency, we want to store least information in memory
        # Hence, we will compute the shifted bins of X and directly reallocate the probability mass to the overlapping Z bins

    for i in range(nbinsX):
        scaled_X_bin_low = X.bins[i] + k
        scaled_X_bin_up = (X.bins[i] + X.bin_width) + k
        if debug:
            print(
                f"Shifting X bin {i} from [{X.bins[i]};{X.bins[i]+X.bin_width}] to [{scaled_X_bin_low};{scaled_X_bin_up}], drop amount={drop}%"
            )

        bin_contribution = get_bin_contribution_to_cpt(
            [scaled_X_bin_low, scaled_X_bin_up], Z.bins, debug=debug
        )
        if debug:
            print(f"i={i}/{nbinsX-1}, j={j}/{nbinsY-1}, z={bin_contribution}")
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
