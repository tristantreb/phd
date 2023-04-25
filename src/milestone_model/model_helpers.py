# Helper functions to create models for belief propagation

import numpy as np
import pandas as pd
import scipy.integrate as integrate
from scipy.stats import norm

# Set global value for tolerance.
# This to account for the rounding error: https://www.cs.drexel.edu/~jpopyack/Courses/CSP/Fa17/extras/Rounding/index.html#:~:text=Rounding%20(roundoff)%20error%20is%20a,word%20size%20used%20for%20integers.
tol_global = 1e-8


## Discretized PDF with the sampling solution
# Let Unblocked FEV1 be a continuous random variable following a uniform distribution between a and b
def get_unblocked_fev1(a, b):
    return np.random.uniform(a, b)


# Let % Small parentAirway blockage be a continuous random variable following a uniform distribution between a and b
def get_small_airway_blockage(a, b):
    return np.random.uniform(a, b)


# Let FEV1 be the product of Unblocked FEV1 and (1 - % Small parentAirway blockage)
# FEV1 is a continuous random variable following a uniform distribution between 0 and 6
def get_fev1(unblocked_fev1, small_airway_blockage):
    # 0% Small parentAirway blockage implies that FEV1 = Unblocked FEV1
    # 100% Small parentAirway blockage implies that FEV1 = 0
    fev1 = unblocked_fev1 * (1 - small_airway_blockage)
    return pd.DataFrame(
        {
            "Unblocked FEV1": [unblocked_fev1],
            "Small airway blockage": [small_airway_blockage],
            "FEV1": [fev1],
        }
    )


# We can generate a sample of FEV1 values and put the results in a dataframe
def generate_fev1_sample(u1, u2, s1, s2, n=10000):
    df = get_fev1(get_unblocked_fev1(u1, u2), get_small_airway_blockage(s1, s2))
    for _ in range(n):
        df = pd.concat(
            [
                df,
                get_fev1(get_unblocked_fev1(u1, u2), get_small_airway_blockage(s1, s2)),
            ]
        )
    return df


## Closed form solution of the PDF
# Let X~U(a,b) and Y~U(c,d) independent.
# This gives the PDF of Z = X*Y
def p_fev1(x, a, b, c, d):
    norm = (b - a) * (d - c)
    if a * d <= c * b:
        if x > a * c and x < a * d:
            return np.log(x / (a * c)) / norm
        elif x >= a * d and x <= c * b:
            return np.log(d / c) / norm
        elif x > c * b and x < b * d:
            return np.log(d * b / x) / norm
        else:
            return 0
    else:
        # exchange X and Y
        return p_fev1(x, c, d, a, b)


## Variable node
class variableNode:
    def __init__(self, name: str, a, b, bin_width, prior={"type": "uniform"}):
        self.tol = tol_global
        self.name = name
        self.a = a
        self.b = b
        self.bin_width = bin_width
        # We're adding bin_width - tol_global to include b in the array of bins so that the last bin can be computed
        # The 1st bin is [a; self.bins[1])
        # The nth bin is [self.bins[n]; self.bins[n+1]]
        # The last bin is [self.bins[-2]; b)
        self.bins = np.arange(a, b + bin_width - self.tol, bin_width)
        self.prior = self.set_prior(self, prior)

    def sample(self):
        return np.random.uniform(self.a, self.b)

    @staticmethod
    def set_prior(self, prior: object):
        if prior["type"] == "uniform":
            proba = self._uniform_prior(self)
        elif prior["type"] == "gaussian":
            proba = self._gaussian_prior(self, prior["mu"], prior["sigma"])
        else:
            raise ValueError(
                f"Prior {prior} not supported. Please use 'uniform' or 'gaussian'."
            )
        # TODO: update to use sum(sum(proba))
        total_proba = sum([sum(p) for p in proba])
        assert (
            total_proba - 1
        ) < self.tol, f"Error computing prior: The sum of the probabilities should be 1, got {total_proba}"
        return proba

    @staticmethod
    def _uniform_prior(self):
        return [[np.array(1 / (len(self.bins) - 1))] for _ in range(len(self.bins) - 1)]

    @staticmethod
    def _gaussian_prior(self, mu: float, sigma: float):
        proba_per_bin = norm.pdf(self.bins[:-1], loc=mu, scale=sigma)
        proba_per_bin_norm = [proba_per_bin / sum(proba_per_bin)]
        return np.transpose(proba_per_bin_norm)


## P(fev1 | unblocked_fev1, small_airway_blockage) can be computed with the closed form solution
# Creates a 3D array from 3 variables
def calc_cpt(
    parentA: variableNode,
    parentB: variableNode,
    C: variableNode,
    tol=tol_global,
    debug=False,
):
    # https://pgmpy.org/factors/discrete.html?highlight=tabular#pgmpy.factors.discrete.CPD.TabularCPD
    nbinsA = len(parentA.bins) - 1
    nbinsB = len(parentB.bins) - 1
    nbinsC = len(C.bins) - 1
    cpt = np.empty((nbinsC, nbinsA, nbinsB))
    print(f"calculating cpt of shape {nbinsC} x {nbinsA} x {nbinsB} (C x A x B) ")

    for i in range(nbinsA):
        # Take a bin in parentA
        b_low = parentA.bins[i]
        b_up = parentA.bins[i + 1]

        for j in range(nbinsB):
            # Take a bin in parentB
            a_low = parentB.bins[j]
            a_up = parentB.bins[j + 1]

            # Get the max possible range of for C=parentA*parentB
            C_min = a_low * b_low
            C_max = a_up * b_up

            total = 0
            abserr = -1
            for c in range(nbinsC):
                # Take a bin in C
                C_low = C.bins[c]
                C_up = C.bins[c + 1]
                # Get the inner intersection of C_range and [C_low, C_up]
                # Compute P(C | parentA, parentB)
                if (
                    (C_min - C_low < tol and C_max - C_low > -tol)
                    or (C_min - C_up < tol and C_max - C_up > -tol)
                    or ((C_min - C_low >= -tol) and (C_max - C_up <= tol))
                ):
                    # The intersection is not empty
                    val, abserr = integrate.quad(
                        p_fev1, C_low, C_up, args=(a_low, a_up, b_low, b_up)
                    )
                    total += val
                    cpt[c, i, j] = val
                    if debug:
                        print(
                            f"idx {c, i, j}, C_low {C_low}, C_up {C_up}, val {val}, C_min {C_min}, C_max {C_max}"
                        )
                else:
                    # The intersection is empty
                    cpt[c, i, j] = 0
                    if debug:
                        print(f"idx {c, i, j} is empty")
            if debug:
                print(f"P(C|U,B) = {cpt[:, i, j]}")
            assert (
                abs(total - 1) < tol
            ), f"Error calculating cpt: The sum of the probabilities should be 1\n Distributions: U({a_low}, {a_up}), B({b_low}, {b_up})\n P(C|U,B) = {cpt[:, i, j]}\n With C range {C_min, C_max}\n For the C bins: {C.bins}\n Abserr = {abserr}"

    return cpt


## P(fev1 | unblocked_fev1, small_airway_blockage) can be computed with the closed form solution
# Creates a 2D array with 3 variables
def calc_pgmpy_cpt(
    parentA: variableNode,
    parentB: variableNode,
    C: variableNode,
    tol=tol_global,
    debug=False,
):
    # https://pgmpy.org/factors/discrete.html?highlight=tabular#pgmpy.factors.discrete.CPD.TabularCPD
    nbinsA = len(parentA.bins) - 1
    nbinsB = len(parentB.bins) - 1
    nbinsC = len(C.bins) - 1
    cpt = np.empty((nbinsC, nbinsA * nbinsB))
    print(f"calculating cpt of shape {nbinsC} x {nbinsA} x {nbinsB} (C x (A x B)) ")

    for i in range(nbinsB):
        # Initialize the index of the current bin in the cpt
        cpt_index = i * (nbinsA - 1)
        if debug:
            print("cpt index", cpt_index)

        # Take a bin in parentA
        b_low = parentB.bins[i]
        b_up = parentB.bins[i + 1]

        for j in range(nbinsA):
            # Take a bin in parentB
            a_low = parentA.bins[j]
            a_up = parentA.bins[j + 1]

            # Get the max possible range of for C=parentA*parentB
            C_min = a_low * b_low
            C_max = a_up * b_up

            total = 0
            abserr = -1
            for c in range(nbinsC):
                # Take a bin in C
                C_low = C.bins[c]
                C_up = C.bins[c + 1]
                # Get the inner intersection of C_range and [C_low, C_up]
                # Compute P(C | parentA, parentB)
                if (
                    (C_min - C_low < tol and C_max - C_low > -tol)
                    or (C_min - C_up < tol and C_max - C_up > -tol)
                    or ((C_min - C_low >= -tol) and (C_max - C_up <= tol))
                ):
                    # The intersection is not empty
                    val, abserr = integrate.quad(
                        p_fev1, C_low, C_up, args=(a_low, a_up, b_low, b_up)
                    )
                    total += val
                    cpt[c, cpt_index + i + j] = val
                    if debug:
                        print(
                            f"idx {c, cpt_index + i + j}, {cpt_index} + {i} + {j}, C_low {C_low}, C_up {C_up}, val {val}, C_min {C_min}, C_max {C_max}"
                        )
                else:
                    # The intersection is empty
                    cpt[c, cpt_index + i + j] = 0
                    if debug:
                        print(f"idx {c, cpt_index + i + j} is empty")
            if debug:
                print(f"P(C|U,B) = {cpt[:, cpt_index + i + j]}")
            assert (
                abs(total - 1) < tol
            ), f"The sum of the probabilities should be 1, got {total}\n Distributions: parentA ~ U({a_low}, {a_up}), parentB ~ U({b_low}, {b_up})\n P(child|parentA,parentB) = {cpt[:, cpt_index + i + j]}\n Range of the child var: [{C_low}; {C_up}]\n Bins of the child: {C.bins}\n Integral abserr = {abserr}"

    return cpt


# Given an observation and an array of bins, this returns the bin that the value falls into
def get_bin_for_value(obs: float, bins: np.array, tol=tol_global):
    # Center bins around value observed
    relative_bins = bins - obs - tol

    # Find the highest negative value of the bins relative to centered bins
    idx = np.where(relative_bins <= 0, relative_bins, -np.inf).argmax()

    lower_idx = bins[idx].item()
    upper_idx = bins[idx + 1].item()
    return ["[{}; {}[".format(lower_idx, upper_idx), idx]
