# Helper functions to create models for belief propagation
import json
from typing import List

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


def name_to_abbr_dict():
    return {
        "Healthy FEV1 (L)": "HFEV1",
        "ecFEV1 (L)": "ecFEV1",
        "ecFEF25-75 % ecFEV1 (%)": "ecFEF25-75%ecFEV1",
        "Airway resistance (%)": "AR",
        "O2 saturation (%)": "O2Sat",
        "Healthy O2 saturation (%)": "HO2Sat",
        "O2 saturation if fully functional alveoli (%)": "O2SatFFA",
        "Inactive alveoli (%)": "IA",
        "Underlying O2 saturation (%)": "UO2Sat",
    }


def name_to_abbr(name: str):
    abbr = name_to_abbr_dict().get(name, "Invalid name")
    if abbr == "Invalid name":
        raise ValueError(f"Invalid name: {name}")

    return abbr


def abbr_to_colname_dict():
    return {
        "HFEV1": "Predicted FEV1",
        "ecFEV1": "ecFEV1",
        "O2Sat": "O2 Saturation",
        "HO2Sat": "Healthy O2 Saturation",
        "ecFEF25-75%ecFEV1": "ecFEF2575%ecFEV1",
        "AR": "AR",
        "IA": "IA",
    }


def abbr_to_colname(name: str):
    colname = abbr_to_colname_dict().get(name, "Invalid abbreviation")
    if colname == "Invalid abbreviation":
        raise ValueError(f"Invalid abbreviation: {name}")

    return colname


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


class DiscreteVariableNode(AbstractVariableNode):
    def __init__(self, name: str, a, b, interval):
        """
        name: variable's name (e.g. "Healthy FEV1 (L)")
        a: lower bound of the variable's domain
        b: upper bound of the variable's domain
        bin_width: width of the bins
        prior: if the prior is None, then the variable is a child variable (and has a CPT)
        """
        self.tol = TOL_GLOBAL
        self.name = name
        self.a = a
        self.b = b
        self.interval = self.interval
        self.values = np.arange(a, b + interval, interval)
        self.card = len(self.bins)


class VariableNode:
    """
    This variable node class can be used to build Bayesian networks as well as factor graphs.

    Use the prior parameter to define the prior of the variable during instanciation. If the variable is a child variable, then the prior should be None.
    Use set_cpt to define the conditional probability table of the variable.
    For simplicity, both prior and cpt then belong to the class's "cpt" attribute
    """

    def __init__(self, name: str, a, b, bin_width, prior):
        """
        name: variable's name (e.g. "Healthy FEV1 (L)")
        a: lower bound of the variable's domain
        b: upper bound of the variable's domain
        bin_width: width of the bins
        prior: if the prior is None, then the variable is a child variable (and has a CPT)
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
        self.card = len(self.bins)
        self.cpt = self.set_prior(prior)

    def get_abbr(self):
        """
        Return the abbreviation of the variable's name
        """
        return name_to_abbr(self.name)

    def get_colname(self):
        """
        Return the column name of the variable's name
        """
        return abbr_to_colname(self.get_abbr())

    def get_bins_str(self):
        """
        bins_str = ["[a, a+bin_width]", "[a+bin_width, a+2*bin_width]", ...
        """
        return list(
            map(lambda x: f"[{round(x,2)}, {round(x + self.bin_width,2)})", self.bins)
        )

    def get_bins_arr(self):
        """
        bins_arr = [[a, a+bin_width], [a+bin_width, a+2*bin_width], ...
        """
        return np.array(
            list(map(lambda x: [x, round(x, 2) + round(self.bin_width, 2)], self.bins))
        )

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
            midbins = np.random.choice(self.midbins, n, p=self.cpt.reshape(-1))

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
        The prior should be a 1D array of probabilities
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

        total_p = sum(p)
        assert (
            total_p - 1
        ) < self.tol, f"Error computing prior: The sum of the probabilities should be 1, got {total_p}"
        return p

    def _uniform_prior(self):
        return np.array([1 / len(self.bins)] * len(self.bins))

    def _gaussian_prior(self, mu: float, sigma: float):
        # print("Defining gaussian prior with mu = {:.2f}, sigma = {}".format(mu, sigma))
        p_arr = norm.pdf(self.bins + self.bin_width / 2, loc=mu, scale=sigma)
        p_arr_norm = p_arr / sum(p_arr)
        return p_arr_norm

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

    def set_cpt(self, cpt):
        """
        Set the conditional probability table of the variable
        Checks that the sum of the probabilities is 1
        """
        total_p = np.sum(cpt, axis=0)
        assert (
            (total_p - 1) < self.tol
        ).all(), f"Error checking CPT: The sum of the probabilities should be 1, got {total_p}"
        self.cpt = cpt

    def get_mean(self, p):
        """
        Returns the distribution's mean given an array of probabilities
        """
        return np.multiply(p, self.midbins).sum()

    def get_mode(self, p):
        """
        Returns the distribution's mode given an array of probabilities
        """
        return self.bins[np.argmax(p)] + self.bin_width / 2

    def get_std(self, p):
        """
        Returns the distribution's standard deviation given an array of probabilities
        """
        mean = self.get_mean(p)
        std = np.sqrt(np.multiply(p, np.power(self.midbins - mean, 2)).sum())
        return std

    def get_skewness(self, p):
        """
        Returns the distribution's skewness given an array of probabilities
        """
        mean = self.get_mean(p)
        std = self.get_std(p)
        skewness = np.multiply(p, np.power(self.midbins - mean, 3)).sum() / np.power(
            std, 3
        )
        return skewness

    def get_bin_for_value(self, obs: float, tol=TOL_GLOBAL):
        """
        Given an observation and an array of bins, this returns the bin that the value falls into
        """
        # Center bins around value observed
        relative_bins = self.bins - obs - tol

        # Find the highest negative value of the bins relative to centered bins
        idx = np.where(relative_bins <= 0, relative_bins, -np.inf).argmax()

        (lower_idx, upper_idx) = self.get_bins_arr()[idx]
        return "[{}; {})".format(lower_idx, upper_idx), idx

    def get_point_message(self, obs_val):
        # Create an array with 1 at the index of the evidence and 0 elsewhere
        message = np.zeros(self.card)
        idx = self.get_bin_for_value(obs_val)[1]
        message[idx] = 1
        return message


class SharedVariableNode(VariableNode):
    """
    In longitudinal models, a shared variable is a variable that is
    connected to a factor belonging to a different time plate.
    The shared variable is able to store messages coming from that factor,
    and operate on them to be used as virtual messages in the slicing belief
    propagation algorithm.
    """

    def __init__(self, name: str, a, b, bin_width, prior):
        super().__init__(name, a, b, bin_width, prior)
        self.name = name
        self.factor_node_key = ""
        self.vmessages = {}
        self.agg_vmessage = np.ones(self.card) / self.card

    def set_factor_node_key(self, factor_node_key):
        """
        Key to identify the factor -> node message in the graph
        """
        self.factor_node_key = factor_node_key

    def add_or_update_message(self, day_key, new_message):
        assert new_message.shape == (
            self.card,
        ), "The message must have the same shape as the variable's cardinality"
        # Always replace the message for that day, even if it already exists
        self.vmessages[day_key] = new_message

    def set_agg_virtual_message(self, vmessage, new_message):
        """
        The new aggregated message is the multiplication of all messages coming from the factor to the node

        Virtual message: multiplication of all factor to node messages excluding current day message
        New message: newly computed factor to node message
        """
        agg_m = np.multiply(vmessage, new_message)
        self.agg_vmessage = agg_m / agg_m.sum()

    def reset(self):
        self.vmessages = {}
        self.agg_vmessage = np.ones(self.card) / self.card

    def get_virtual_message(self, day_key, agg_method=True):
        """
        Returns the aggregated message, excluding the message from the current day
        if applicable (if n_epoch > 0).
        """
        if agg_method:
            agg_m = self.agg_vmessage

            if day_key not in self.vmessages.keys():
                return agg_m

            # Remove previous today's message from agg_m
            curr_m = self.vmessages[day_key]
            agg_m_excl_curr_m = np.divide(
                agg_m, curr_m, out=np.zeros_like(agg_m), where=curr_m != 0
            )
            return agg_m_excl_curr_m / agg_m_excl_curr_m.sum()

        # Multiply all messages together (less efficient)
        # Remove message with day_key from the list of messages
        vmessages = self.vmessages.copy()
        if day_key in self.vmessages.keys():
            vmessages.pop(day_key)

        if len(vmessages) == 0:
            return None
        elif len(vmessages) == 1:
            return list(vmessages.values())[0]
        else:
            agg_message = np.ones(self.card)
            for vm in vmessages.values():
                agg_message = np.multiply(agg_message, vm)
                # Renormalise each time to avoid numerical issues (message going to 0)
                agg_message = agg_message / agg_message.sum()
            return agg_message


def encode_node_variable(var):
    """
    We must encode variables to JSON format to share them between Dash's callbacks
    """
    var_as_dict = vars(var)
    var_as_dict = {k: var_as_dict[k] for k in ("name", "a", "b", "bin_width", "cpt")}
    # Transform ndarray to list
    var_as_dict["cpt"] = var_as_dict["cpt"].tolist()
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
    class_var = VariableNode(name, a, b, bin_width, prior=None)
    # Transform list to ndarray
    class_var.cpt = np.array(jsoned_var["cpt"])
    return class_var


def calc_pgmpy_cpt_X_x_1_minus_Y(
    X: VariableNode,
    Y: VariableNode,
    Z: VariableNode,
    tol=TOL_GLOBAL,
    debug=False,
):
    """
    Function specific to Z = X*(1-Y)
    Y bins must be given in %, not in decimals from the interval [0,1]
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
        (a_low, a_up) = X.get_bins_arr()[i]
        for j in range(nbinsY):
            # Take a bin in Y
            (b_low, b_up) = Y.get_bins_arr()[j] / 100
            # Get the max possible range of for C=X*Y
            Z_min = a_low * (1 - b_up)
            Z_max = a_up * (1 - b_low)
            total = 0
            abserr = -1
            for z in range(nbinsZ):
                # Take a bin in C
                (Z_low, Z_up) = Z.get_bins_arr()[z]
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


def get_intersecting_bins_idx(intersecting_bin, bin_list, debug=False):
    """
    Returns the indices of the bins in bin_list that intersect with bin
    Assumes that the bins are sorted in increasing order and with constant bin_width
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

    left_away_error = ValueError("A portion of the scaled X bin is outside the Z bins")
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
