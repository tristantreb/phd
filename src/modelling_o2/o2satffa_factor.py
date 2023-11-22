import numpy as np

import src.models.helpers as mh


def drop_func(x):
    drop_params = np.array(
        [3.89754850e01, 1.00902396e02, -2.04542149e-01, 1.57422295e-02, -4.00994278e-04]
    )

    x0 = drop_params[0]
    y0 = 0
    k1 = -drop_params[2]
    k2 = -drop_params[3]
    k3 = -drop_params[4]

    return np.piecewise(
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


def calc_cpt_O2SatFFA_HO2Sat_AR(
    O2SatFFA: mh.variableNode, HO2Sat: mh.variableNode, AR: mh.variableNode, debug=False
):
    """
    Computes the CPT for P(O2SatFFA|HO2Sat, AR)
    """

    cpt = calc_cpt_X_minus_funcY(O2SatFFA, HO2Sat, AR, drop_func, debug=debug)
    return cpt.reshape(len(O2SatFFA.bins), len(HO2Sat.bins) * len(AR.bins))


def calc_cpt_X_minus_funcY(
    Z: mh.variableNode,
    X: mh.variableNode,
    Y: mh.variableNode,
    func,
    tol=1e-6,
    debug=False,
):
    """
    Computes the CPT for P(Z|X, Y), when Z is shifted from X by a function of Y
    Z = X - f(Y)
    X: parent variable
    Y: drop variable - the drop amount is given by f(Y) = X - Z
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
    nbinsX = len(X.bins)
    nbinsY = len(Y.bins)
    nbinsZ = len(Z.bins)

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
            shifted_X_bin_low = X.bins[i] - drop
            shifted_X_bin_up = X.bins[i] + X.bin_width - drop
            if debug:
                print(
                    f"Shifting X bin {i} from [{X.bins[i]};{X.bins[i]+X.bin_width}] to [{shifted_X_bin_low};{shifted_X_bin_up}], drop amount={drop}"
                )

            bin_contribution = get_bin_contribution_to_cpt(
                [shifted_X_bin_low, shifted_X_bin_up], Z.bins, debug=debug
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


def get_intersecting_bins_idx(intersecting_bin, bin_list, debug=False):
    """
    Returns the indices of the bins in bin_list that intersect with bin
    Assumes that the bins are sorted in increasing order and with same bin_width
    """
    bin_list_width = bin_list[1] - bin_list[0]
    # Width of intersecting bin can't be 0
    if intersecting_bin[1] - intersecting_bin[0] == 0:
        raise ValueError("Intersecting bin width should be non-zero")

    # Checks that the bin falls strictly within the range [bin_list[0]; bin_list[-1]]
    overlap_error = ValueError(
        f"Shifted X bin {intersecting_bin} intersects none of the Z bins {bin_list}"
    )
    if intersecting_bin[1] < bin_list[0]:
        raise overlap_error
    # Raise if X_shifted_bin_low is larger than the largest Z bin
    if intersecting_bin[0] >= bin_list[-1] + bin_list_width:
        raise overlap_error

    left_away_error = ValueError("A portion of the shifted X bin is outside the Z bins")
    if intersecting_bin[0] < bin_list[0]:
        raise left_away_error
    if intersecting_bin[1] > bin_list[-1] + bin_list_width:
        raise left_away_error

    idx1, idxn = np.searchsorted(bin_list, [intersecting_bin[0], intersecting_bin[1]])
    if debug:
        print("Idx from searchsorted func", idx1, ",", idxn)

    k1 = idx1 - 1
    if idx1 <= len(bin_list) - 1 and bin_list[idx1] == intersecting_bin[0]:
        k1 = idx1

    kn = idxn - 1

    if debug:
        print(
            "Bins idx",
            k1,
            ",",
            kn,
            "corresponding to values",
            bin_list[k1],
            bin_list[kn],
        )
    return (k1, kn)


def get_bin_contribution_to_cpt(shifted_X_bin, Z_bins, debug=False, tol=1e-6):
    """
    Returns a vector of probabilities representing the contribution of the shifted X bin to a set of Z_bins (i.e. cpt[:, i, j])

    Each element in Z_bins correspond to the interval [Z_bins[i]; Z_bins[i] + Z_bins_width]
    """

    shifted_X_binwidth = shifted_X_bin[1] - shifted_X_bin[0]
    if shifted_X_binwidth == 0:
        raise ValueError("Shifted X bin width should be non-zero")
    Z_bins_width = Z_bins[1] - Z_bins[0]

    p_vect = np.zeros(len(Z_bins))
    (k1, kn) = get_intersecting_bins_idx(shifted_X_bin, Z_bins)

    for k in range(k1, kn + 1):
        # Find the overlapping region between the shifted X bin and the Z bin
        # The overlapping region is the intersection of the two intervals
        # The intersection is the max of the lower bounds and the min of the upper bounds
        # The intersection is empty if the lower bound is larger than the upper bound
        intersection = max(shifted_X_bin[0], Z_bins[k]), min(
            shifted_X_bin[1], Z_bins[k] + Z_bins_width
        )

        # The probability mass is the length of the intersection divided by the length of the shifted X bin
        p = (intersection[1] - intersection[0]) / shifted_X_binwidth
        p_vect[k] = p

        if debug:
            print(
                f"k={k}, bin=[{Z_bins[k]};{Z_bins[k] + Z_bins_width}], p={p} (={intersection[1] - intersection[0]}/{shifted_X_binwidth})"
            )
    # Raise if sum of probabilities is larger than 1
    total = np.sum(p_vect)
    assert (
        abs(total - 1) < tol
    ), f"The sum of the probabilities should be 1, got {total}\np_vect={p_vect}"

    return p_vect
