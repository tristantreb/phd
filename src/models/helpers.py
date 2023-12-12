# Helper functions to create models for belief propagation
import json

import numpy as np
import pandas as pd
import scipy.integrate as integrate
from scipy.stats import norm

import src.modelling_fev1.pred_fev1 as pred_fev1
import src.modelling_o2.ho2sat as ho2sat

# Set global value for tolerance.
# This to account for the rounding error: https://www.cs.drexel.edu/~jpopyack/Courses/CSP/Fa17/extras/Rounding/index.html#:~:text=Rounding%20(roundoff)%20error%20is%20a,word%20size%20used%20for%20integers.
TOL_GLOBAL = 1e-6
# Switch from 1e-8 to 1e-6 to because got 0.9999999885510139 sum of probabilities for a model with AW


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
def generate_fev1_sample(u1, u2, b1, b2, n=10000):
    df = get_fev1(get_unblocked_fev1(u1, u2), get_small_airway_blockage(b1, b2))
    for _ in range(n):
        df = pd.concat(
            [
                df,
                get_fev1(get_unblocked_fev1(u1, u2), get_small_airway_blockage(b1, b2)),
            ]
        )
    return df


def PDF_X_x_Y(z, a, b, c, d):
    """
    Let X~U(a,b) and Y~U(c,d) independent.
    This function is the closed form solution of the PDF of Z = X*Y
    It returns f_Z(z)
    """
    norm = (b - a) * (d - c)
    if a * d <= c * b:
        if z > a * c and z < a * d:
            return np.log(z / (a * c)) / norm
        elif z >= a * d and z <= c * b:
            return np.log(d / c) / norm
        elif z > c * b and z < b * d:
            return np.log(d * b / z) / norm
        else:
            return 0
    else:
        # exchange X and Y
        return PDF_X_x_Y(z, c, d, a, b)


def PDF_X_x_1_minus_Y(z, x_a, x_b, y_a, y_b):
    """
    Let X~U(a,b) and Y~U(c,d) independent.
    This function is the closed form PDF for Z = X * (1-Y)
    It returns f_Z(z)

    Input ordering matters!!
    """
    if 1 - y_b < 0 or 1 - y_b > 1:
        # if 1 - y_b < 0 or 1 - y_b > 1:
        raise ValueError(f"y_b should be between 0 and 1, got {y_b}")
    if 1 - y_a < 0 or 1 - y_a > 1:
        # if 1 - y_a < 0 or 1 - y_a > 1:
        raise ValueError(f"y_a should be between 0 and 1, got {y_a}")
    return PDF_X_x_Y(z, x_a, x_b, 1 - y_b, 1 - y_a)


## Variable node
class variableNode:
    def __init__(self, name: str, a, b, bin_width, prior):
        self.tol = TOL_GLOBAL
        self.name = name
        self.a = a
        self.b = b
        self.bin_width = bin_width
        # Since we define the bin by its lower bound, the last bin is [self.bins[-1]; b), b should be excluded from the array of bins
        # The 1st bin is [a; self.bins[1])
        # The nth bin is [self.bins[n]; self.bins[n+1]]
        # The last bin is [self.bins[-1]; b)
        # We're b - TOL_GLOBAL allows to exclude b from the array of bins
        # self.bins = np.arange(a, b + bin_width - self.tol, bin_width)
        self.bins = np.arange(a, b - self.tol, bin_width)
        self.bins_arr = np.array(
            list(map(lambda x: [x, round(x, 2) + round(self.bin_width, 2)], self.bins))
        )
        self.bins_str = list(
            map(lambda x: f"[{round(x,2)}, {round(x + self.bin_width,2)})", self.bins)
        )
        # Sets prior or CPT
        self.prior = self.set_prior(prior)

    def sample(self):
        return np.random.uniform(self.a, self.b)

    @classmethod
    def set_prior(self, prior):
        """
        The prior of node variable is a 2D array of shape (len(bins), 1), so that: sum(P(nodeVariable)) = 1
        That is because in PGMPY, a prior is essentially a CPT with only one parent state
        """
        # Child variables have no prior
        if prior == None:
            return None
        # Node variable specific priors
        elif prior["type"] == "default":
            if self.name == "Healthy FEV1 (L)":
                height = prior["height"]
                age = prior["age"]
                sex = prior["sex"]
                p = pred_fev1.calc_hfev1_prior(
                    self.bins + self.bin_width / 2, height, age, sex
                )
            elif self.name == "Healthy O2 Saturation (%)":
                height = prior["height"]
                sex = prior["sex"]
                params = ho2sat.calc_healthy_O2_sat(height, sex)
                p = self._gaussian_prior(params["mean"], params["sigma"])
        # General priors
        elif prior["type"] == "uniform":
            p = self._uniform_prior(self)
        elif prior["type"] == "gaussian":
            p = self._gaussian_prior(prior["mu"], prior["sigma"])
        elif prior["type"] == "uniform + gaussian tail":
            p = self._uniform_prior_with_gaussian_tail(
                prior["constant"], prior["sigma"]
            )
        else:
            raise ValueError(f"Prior for {self.name} not recognized")

        total_p = sum(sum(p))
        assert (
            total_p - 1
        ) < self.tol, f"Error computing prior: The sum of the probabilities should be 1, got {total_p}"
        return p

    @classmethod
    def _uniform_prior(self):
        return np.array([1 / len(self.bins)] * len(self.bins)).reshape(
            len(self.bins), 1
        )

    @classmethod
    def _gaussian_prior(self, mu: float, sigma: float):
        print("Defining gaussian prior with mu = {:.2f}, sigma = {}".format(mu, sigma))
        p_arr = norm.pdf(self.bins + self.bin_width / 2, loc=mu, scale=sigma)
        p_arr_norm = [p_arr / sum(p_arr)]
        return np.transpose(p_arr_norm)

    @classmethod
    def _uniform_prior_with_gaussian_tail(self, constant: float, sigma: float):
        u_prior = self._uniform_prior(self)
        g_prior = self._gaussian_prior(self, constant, sigma)

        # Get the index of the bin that contains the constant
        idx = np.where(self.bins <= constant, self.bins, -np.inf).argmax()
        print(
            f"Defining uniform prior until {round(self.bins[idx]+self.bin_width,2)} L, then gaussian tail up to {self.bins[-1]} L"
        )

        # Use the uniform prior for indices <= idx, else use the gaussian prior
        u_prior_norm = u_prior[:idx] / sum(np.array(u_prior[:idx]))
        g_prior_norm = g_prior[idx:] / sum(np.array(g_prior[idx:]))

        p_arr = np.concatenate([u_prior_norm, g_prior_norm], axis=0)

        # Renormalise
        p_arr = p_arr / sum(p_arr)
        return p_arr

    @classmethod
    def get_mean(self, p):
        """
        Returns the distribution's mean given an array of probabilities
        """
        return np.multiply(p, self.bins + self.bin_width / 2).sum()

    @classmethod
    def get_mode(self, p):
        """
        Returns the distribution's mode given an array of probabilities
        """
        return self.bins[np.argmax(p)] + self.bin_width / 2


# ## P(fev1 | unblocked_fev1, small_airway_blockage) can be computed with the closed form solution
# # Creates a 3D array from 3 variables
# def calc_cpt(
#     parentA: variableNode,
#     parentB: variableNode,
#     C: variableNode,
#     tol=TOL_GLOBAL,
#     debug=False,
# ):
#     # https://pgmpy.org/factors/discrete.html?highlight=tabular#pgmpy.factors.discrete.CPD.TabularCPD
#     nbinsA = len(parentA.bins)
#     nbinsB = len(parentB.bins)
#     nbinsC = len(C.bins)
#     cpt = np.empty((nbinsC, nbinsA, nbinsB))
#     # print(f"calculating cpt of shape {nbinsC} x {nbinsA} x {nbinsB} (C x A x B) ")

#     for i in range(nbinsA):
#         # Take a bin in parentA
#         b_low = parentA.bins[i]
#         b_up = parentA.bins[i + 1]

#         for j in range(nbinsB):
#             # Take a bin in parentB
#             a_low = parentB.bins[j]
#             a_up = parentB.bins[j + 1]

#             # Get the max possible range of for C=parentA*parentB
#             C_min = a_low * b_low
#             C_max = a_up * b_up

#             total = 0
#             abserr = -1
#             for c in range(nbinsC):
#                 # Take a bin in C
#                 C_low = C.bins[c]
#                 C_up = C.bins[c + 1]
#                 # Get the inner intersection of C_range and [C_low, C_up]
#                 # Compute P(C | parentA, parentB)
#                 if (
#                     (C_min - C_low < tol and C_max - C_low > -tol)
#                     or (C_min - C_up < tol and C_max - C_up > -tol)
#                     or ((C_min - C_low >= -tol) and (C_max - C_up <= tol))
#                 ):
#                     # The intersection is not empty
#                     val, abserr = integrate.quad(
#                         PDF_X_x_1_minus_Y, C_low, C_up, args=(a_low, a_up, b_low, b_up)
#                     )
#                     total += val
#                     cpt[c, i, j] = val
#                     if debug:
#                         print(
#                             f"idx {c, i, j}, C_low {C_low}, C_up {C_up}, val {val}, C_min {C_min}, C_max {C_max}"
#                         )
#                 else:
#                     # The intersection is empty
#                     cpt[c, i, j] = 0
#                     if debug:
#                         print(f"idx {c, i, j} is empty")
#             if debug:
#                 print(f"P(C|U,B) = {cpt[:, i, j]}")
#             assert (
#                 abs(total - 1) < tol
#             ), f"Error calculating cpt: The sum of the probabilities should be 1\n Distributions: U({a_low}, {a_up}), B({b_low}, {b_up})\n P(C|U,B) = {cpt[:, i, j]}\n With C range {C_min, C_max}\n For the C bins: {C.bins}\n Abserr = {abserr}"

#     return cpt


def calc_pgmpy_cpt_X_x_1_minus_Y(
    X: variableNode,
    Y: variableNode,
    Z: variableNode,
    tol=TOL_GLOBAL,
    debug=False,
):
    """
    Function specific to Z = X*(1-Y)
    P(fev1 | unblocked_fev1, small_airway_blockage) can be computed with the closed form solution
    Creates a 2D array with 3 variables
    """
    # https://pgmpy.org/factors/discrete.html?highlight=tabular#pgmpy.factors.discrete.CPD.TabularCPD
    nbinsX = len(X.bins)
    nbinsY = len(Y.bins)
    nbinsZ = len(Z.bins)
    cpt = np.empty((nbinsZ, nbinsX * nbinsY))
    # print(f"calculating cpt of shape {nbinsZ} x {nbinsX} x {nbinsY} (C x (A x B)) ")

    for i in range(nbinsX):
        # Initialize the index of the current bin in the cpt
        cpt_index = i * (nbinsY - 1)
        if debug:
            print("cpt index", cpt_index)

        # Take a bin in X
        (a_low, a_up) = X.bins_arr[i]

        for j in range(nbinsY):
            # Take a bin in Y
            (b_low, b_up) = Y.bins_arr[j] / 100

            # Get the max possible range of for C=X*Y
            Z_min = a_low * (1 - b_up)
            Z_max = a_up * (1 - b_low)

            total = 0
            abserr = -1
            for z in range(nbinsZ):
                # Take a bin in C
                (Z_low, Z_up) = Z.bins_arr[z]
                # Get the inner intersection of Z_range and [Z_low, Z_up]
                # Compute P(C | X, Y)
                if (
                    (Z_min - Z_low < tol and Z_max - Z_low > -tol)
                    or (Z_min - Z_up < tol and Z_max - Z_up > -tol)
                    or ((Z_min - Z_low >= -tol) and (Z_max - Z_up <= tol))
                ):
                    # The intersection is not empty
                    val, abserr = integrate.quad(
                        PDF_X_x_1_minus_Y, Z_low, Z_up, args=(a_low, a_up, b_low, b_up)
                    )
                    total += val
                    cpt[z, cpt_index + i + j] = val
                    if debug:
                        print(
                            f"idx {z, cpt_index + i + j}, {cpt_index} + {i} + {j}, Z_low {Z_low}, Z_up {Z_up}, val {val}, Z_min {Z_min}, Z_max {Z_max}"
                        )
                else:
                    # The intersection is empty
                    cpt[z, cpt_index + i + j] = 0
                    if debug:
                        print(f"idx {z, cpt_index + i + j} is empty")
            if debug:
                print(f"P(Z|U,B) = {cpt[:, cpt_index + i + j]}")
            assert (
                abs(total - 1) < tol
            ), f"The sum of the probabilities should be 1, got {total}\nDistributions: {X.name} ~ U({a_low}, {a_up}), {Y.name} ~ U({b_low}, {b_up})\nRange over the child bins = [{Z_min}; {Z_max})\nP({Z.name}|{X.name}, {Y.name}) = {cpt[:, cpt_index + i + j]}\n {Z.name} bins: {Z.bins}\n Integral abserr = {abserr}"

    return cpt


def encode_node_variable(var):
    """
    We must encode variables to JSON format to share them between Dash's callbacks
    """
    var_as_dict = vars(var)
    var_as_dict = {k: var_as_dict[k] for k in ("name", "a", "b", "bin_width", "prior")}
    # Transform ndarray to list
    var_as_dict["prior"] = var_as_dict["prior"].tolist()
    return json.dumps(var_as_dict)


def decode_node_variable(jsoned_var):
    """
    Decoding variables from JSON format
    """
    # From json to dict
    jsoned_var = json.loads(jsoned_var)
    name = jsoned_var["name"]
    a = jsoned_var["a"]
    b = jsoned_var["b"]
    bin_width = jsoned_var["bin_width"]
    class_var = variableNode(name, a, b, bin_width, prior=None)
    # Transform list to ndarray
    class_var.prior = np.array(jsoned_var["prior"])
    return class_var
