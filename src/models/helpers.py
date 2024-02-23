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
        """
        name: variable's name (e.g. "Healthy FEV1 (L)")
        a: lower bound of the variable's domain
        b: upper bound of the variable's domain
        bin_width: width of the bins
        prior: variable's prior distribution, possible None if it's a cpt
        """
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
        self.bins = np.arange(a, b - self.tol, bin_width)
        self.midbins = self.bins + self.bin_width / 2
        # bins_arr = [[a, a+bin_width], [a+bin_width, a+2*bin_width], ...
        self.bins_arr = np.array(
            list(map(lambda x: [x, round(x, 2) + round(self.bin_width, 2)], self.bins))
        )
        # bins_str = ["[a, a+bin_width]", "[a+bin_width, a+2*bin_width]", ...
        self.bins_str = list(
            map(lambda x: f"[{round(x,2)}, {round(x + self.bin_width,2)})", self.bins)
        )
        # Sets prior or CPT
        self.prior = self.set_prior(prior)

    def sample(self, n=1, p=None):
        """
        Randomly select a midbins from the variable prior's distribution
        If the variable was continuous but has been discretised, it returns a random value inside the sampled bins range
        If the variable is inherently dicrete (e.g. O2 saturation), it returns the midbin as it is the actual real value

        By default it uses the variable's prior, but it can also use a custom distribution p
        """
        if p is not None:
            midbins = np.random.choice(self.midbins, n, p=p.reshape(-1))
        else:
            midbins = np.random.choice(self.midbins, n, p=self.prior.reshape(-1))

        # If the variable is inherently discrete, return the midbins
        if self.name == "O2 saturation (%)":
            return midbins

        def sample_from_bin(bin):
            """
            When using a continuour variable discretised with bins, we sample from the bin
            """
            return np.random.uniform(bin - self.bin_width / 2, bin + self.bin_width / 2)

        # Otherwise sample from the bins
        sample = np.array(list(map(sample_from_bin, midbins)))
        return sample

    def get_distribution_as_sample(self, p, p_threshold=0.01, print_sample_size=True):
        """
        Creates a sample of n values that reflects the distribution best
        Controle the sample size with p_threshold
        """
        n_vals_per_bin_arr = p / p_threshold
        n_vals_per_bin_arr = np.round(n_vals_per_bin_arr)
        n_vals_per_bin_arr = n_vals_per_bin_arr.astype(int)

        if print_sample_size:
            print(f"Sampling {sum(n_vals_per_bin_arr)} values from {self.name}")

        bin_vals = np.repeat(self.bins, n_vals_per_bin_arr)
        return bin_vals

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
            elif self.name == "Healthy O2 saturation (%)":
                height = prior["height"]
                sex = prior["sex"]
                params = ho2sat.calc_healthy_O2_sat(height, sex)
                p = self._gaussian_prior(params["mean"], params["sigma"])
            else:
                raise ValueError(f"Prior for {self.name} not recognized")
        # General priors
        elif prior["type"] == "uniform":
            p = self._uniform_prior()
        elif prior["type"] == "gaussian":
            p = self._gaussian_prior(prior["mu"], prior["sigma"])
        elif prior["type"] == "uniform + gaussian tail":
            p = self._uniform_prior_with_gaussian_tail(
                prior["constant"], prior["sigma"]
            )
        elif prior["type"] == "custom":
            p = prior["p"]
        else:
            raise ValueError(f"Prior for {self.name} not recognized")

        total_p = sum(sum(p))
        assert (
            total_p - 1
        ) < self.tol, f"Error computing prior: The sum of the probabilities should be 1, got {total_p}"
        return p

    def _uniform_prior(self):
        return np.array([1 / len(self.bins)] * len(self.bins)).reshape(
            len(self.bins), 1
        )

    def _gaussian_prior(self, mu: float, sigma: float):
        # print("Defining gaussian prior with mu = {:.2f}, sigma = {}".format(mu, sigma))
        p_arr = norm.pdf(self.bins + self.bin_width / 2, loc=mu, scale=sigma)
        p_arr_norm = [p_arr / sum(p_arr)]
        return np.transpose(p_arr_norm)

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

    def get_mean(self, p):
        """
        Returns the distribution's mean given an array of probabilities
        """
        return np.multiply(p, self.bins + self.bin_width / 2).sum()

    def get_mode(self, p):
        """
        Returns the distribution's mode given an array of probabilities
        """
        return self.bins[np.argmax(p)] + self.bin_width / 2

    def get_bin_for_value(self, obs: float, tol=TOL_GLOBAL):
        """
        Given an observation and an array of bins, this returns the bin that the value falls into
        """
        # Center bins around value observed
        relative_bins = self.bins - obs - tol

        # Find the highest negative value of the bins relative to centered bins
        idx = np.where(relative_bins <= 0, relative_bins, -np.inf).argmax()

        (lower_idx, upper_idx) = self.bins_arr[idx]
        return "[{}; {})".format(lower_idx, upper_idx), idx


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
