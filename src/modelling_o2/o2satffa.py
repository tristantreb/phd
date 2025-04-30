import numpy as np

import models.helpers as mh


def multiplicative_drop_func(x):
    """
    HO2SatFFA = HO2Sat * drop_func(AR)
    The top envelope was parametrised in 2023-11-07_breathe_O2_modelling.ipynb > fit top envelope
    The top envelope only gives the shape, it does not give the vertical position.
    """
    drop_params = np.array(
        [3.50714590e01, 1.00849203e02, -1.13416141e-01, 5.74567744e-03, -1.21470421e-04]
    )

    x0 = drop_params[0]
    # For small AR (<x0), there is no drop in HO2Sat, hence, the vertical position y0 should be set to 100%.
    y0 = 100
    k1 = drop_params[2]
    k2 = drop_params[3]
    k3 = drop_params[4]

    drop = np.piecewise(
        x,
        [x <= x0],
        [
            lambda x: y0,
            lambda x: k3 * np.power((x - x0), 3)
            + k2 * np.power((x - x0), 2)
            + k1 * (x - x0)
            + y0,
        ],
    )
    return drop / 100


def calc_cpt(
    O2SatFFA: mh.VariableNode, HO2Sat: mh.VariableNode, AR: mh.VariableNode, debug=False
):
    """
    Computes the CPT for P(O2SatFFA|HO2Sat, AR)
    """

    cpt = calc_cpt_X_x_funcY(
        O2SatFFA, HO2Sat, AR, multiplicative_drop_func, debug=debug
    )
    return cpt.reshape(O2SatFFA.card, HO2Sat.card, AR.card)
    # return cpt.reshape(O2SatFFA.card, HO2Sat.card * AR.card)


def calc_cpt_X_x_funcY(
    Z: mh.VariableNode,
    X: mh.VariableNode,
    Y: mh.VariableNode,
    func,
    tol=1e-6,
    debug=False,
):
    """
    Computes the CPT for P(Z|X, Y), when Z is scaled from X by a function of Y
    Z = X * f(Y)
    X: parent variable
    Y: drop variable - the drop amount is given by f(Y) = Z/X
    Z: child variable
    func: must be a monotonous continuous function that has values in Y and returns values in Z

    We compute the CPT with a shift and conquer method:
    1) Start with a CPT zeroed out probabilities
    2) Shift all X bin intervals by the drop amount
    3) For each shifted X bin, evenly spread the X bin  onto the overlapping Z bins
    4) Normalise the CPT

    This allows the function to be agnostic of how X and Z are binned.

    - What happens when the function is shifted outside the boundary? -> Raise an error as it shouldn't happen by how the model is built
    """
    nbinsX = X.card
    nbinsY = Y.card
    nbinsZ = Z.card

    cpt = np.zeros([nbinsZ, nbinsX, nbinsY])
    if debug:
        print(f"Shape of cpt: {cpt.shape}")

    for j in range(nbinsY):
        # The drop amount for a Y bin is the average of the function over the bin
        # Simplified to avg( f(bin_up) - f(bin_low) )
        drop = np.mean([func(Y.bins[j] + Y.bin_width), func(Y.bins[j])])
        if debug:
            print(f"Drop for Y bin {j} ([{Y.bins[j]};{Y.bins[j]+Y.bin_width}]): {drop}")

        # For computational efficiency, we want to store least information in memory
        # Hence, we will compute the shifted bins of X and directly reallocate the probability mass to the overlapping Z bins

        for i in range(nbinsX):
            scaled_X_bin_low = X.bins[i] * drop
            scaled_X_bin_up = (X.bins[i] + X.bin_width) * drop
            if debug:
                print(
                    f"Shifting X bin {i} from [{X.bins[i]};{X.bins[i]+X.bin_width}] to [{scaled_X_bin_low};{scaled_X_bin_up}], drop amount={drop}%"
                )

            bin_contribution = mh.get_bin_contribution_to_cpt(
                [scaled_X_bin_low, scaled_X_bin_up], Z.bins, debug=debug
            )
            if debug:
                print(f"i={i}/{nbinsX-1}, j={j}/{nbinsY-1}, z={bin_contribution}")
            cpt[:, i, j] += bin_contribution

        # Normalise all cpt(:, i, j) to 1
        total = np.sum(cpt[:, i, j])
        if debug:
            print(f"Results before normalisation sum(cpt[:, {i}, {j}])={total}")
        cpt[:, i, j] /= total

        # Raise if sum of probabilities is larger than 1
        total = np.sum(cpt[:, i, j])
        assert (
            abs(total - 1) < tol
        ), f"The sum of the probabilities should be 1, got sum(cpt[:, {i}, {j}])={total}])"

    return cpt
