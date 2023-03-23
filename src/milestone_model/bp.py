import numpy as np
import pandas as pd
import scipy.integrate as integrate


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
    def __init__(self, name: str, a, b, bin_width):
        self.name = name
        self.a = a
        self.b = b
        self.bin_width = bin_width
        self.bins = np.arange(a, b + bin_width, bin_width)

    def sample(self):
        return np.random.uniform(self.a, self.b)

    @staticmethod
    def marginal(self):
        return [[self.bin_width / (self.b - self.a)] for _ in range(len(self.bins) - 1)]


## P(fev1 | unblocked_fev1, small_airway_blockage) can be computed with the closed form solution
def calc_cpt(parentA: variableNode, parentB: variableNode, C: variableNode):
    cpt = np.nan((len(parentA.bins) - 1, len(parentB.bins) - 1, len(C.bins) - 1))

    for i in range(len(parentA.bins) - 1):
        # Take a bin in  parentA
        a_low = parentA.bins[i]
        a_up = parentA.bins[i + 1]

        for j in range(len(parentB.bins) - 1):
            # Take a bin in  parentB
            b_low = parentB.bins[j]
            b_up = parentB.bins[j + 1]

            # Get the max possible range of for C=parentA*parentB
            C_range = np.array([a_low * b_low, a_up * b_up])

            for k in range(len(C.bins) - 1):
                # Take a bin in  C
                C_low = C.bins[k]
                C_up = C.bins[k + 1]

                # Get the inner intersection of C_range and [C_low, C_up]
                # Compute P(C | parentA, parentB)
                if (C_range[0] >= C_low and C_range[0] < C_up) or (
                    C_range[1] > C_low and C_range[1] <= C_up
                ):
                    # The intersection is not empty
                    val, abserr = integrate.quad(
                        p_fev1, C_low, C_up, args=(a_low, a_up, b_low, b_up)
                    )
                    cpt[i, j, k] = val
                else:
                    # The intersection is empty
                    cpt[i, j, k] = 0
    return cpt


## P(fev1 | unblocked_fev1, small_airway_blockage) can be computed with the closed form solution
def calc_pgmpy_cpt(parentA: variableNode, parentB: variableNode, C: variableNode):
    # https://pgmpy.org/factors/discrete.html?highlight=tabular#pgmpy.factors.discrete.CPD.TabularCPD
    cpt = np.empty((len(C.bins) - 1, (len(parentB.bins) - 1) * (len(parentA.bins) - 1)))

    for i in range(len(parentB.bins) - 1):
        # Take a bin in parentA
        b_low = parentB.bins[i]
        b_up = parentB.bins[i + 1]

        for j in range(len(parentA.bins) - 1):
            # Take a bin in parentB
            a_low = parentA.bins[j]
            a_up = parentA.bins[j + 1]

            # Get the max possible range of for C=parentA*parentB
            C_range = np.array([a_low * b_low, a_up * b_up])

            for c in range(len(C.bins) - 1):
                # Take a bin in C
                C_low = C.bins[c]
                C_up = C.bins[c + 1]

                # Get the inner intersection of C_range and [C_low, C_up]
                # Compute P(C | parentA, parentB)
                if (C_range[0] >= C_low and C_range[0] < C_up) or (
                    C_range[1] > C_low and C_range[1] <= C_up
                ):
                    # The intersection is not empty
                    val, abserr = integrate.quad(
                        p_fev1, C_low, C_up, args=(a_low, a_up, b_low, b_up)
                    )
                    cpt[c, 2 * i + j] = round(val, 3)
                else:
                    # The intersection is empty
                    cpt[c, 2 * i + j] = 0
    return cpt
