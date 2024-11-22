# Helper functions to create models for belief propagation
import json
from datetime import datetime
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
        "Underlying ecFEV1 (L)": "uecFEV1",
        "ecFEF25-75 % ecFEV1 (%)": "ecFEF25-75%ecFEV1",
        "Airway resistance (%)": "AR",
        # "Airway resistance": "AR",
        "O2 saturation (%)": "O2Sat",
        "Healthy O2 saturation (%)": "HO2Sat",
        "O2 saturation if fully functional alveoli (%)": "O2SatFFA",
        "Inactive alveoli (%)": "IA",
        "Underlying O2 saturation (%)": "uO2Sat",
        "Days elapsed": "DE",
    }


def name_to_abbr(name: str):
    elapsed = False
    if "day" in name:
        elapsed = True
        # Then it's a variable that spans over time
        name, suffix = str.split(name, " day ")
        day_n, unit = suffix.split(" (")
        # unit = unit[0]
    abbr = name_to_abbr_dict().get(name, "Invalid name")
    if abbr == "Invalid name":
        raise ValueError(f"Invalid name: {name}")
    # if elapsed:
    #     abbr = f"{abbr}{day_n}"

    return abbr


def abbr_to_name_dict():
    # inverse of name_to_abbr_dict
    vals = name_to_abbr_dict().keys()
    keys = name_to_abbr_dict().values()
    return dict(zip(keys, vals))


def abbr_to_name(abbr: str):
    # inverse of name_to_abbr
    name = abbr_to_name_dict().get(abbr, "Invalid abbreviation")
    if name == "Invalid abbreviation":
        raise ValueError(f"Invalid abbreviation: {abbr}")
    return name


def abbr_to_colname_dict():
    return {
        "HFEV1": "Healthy FEV1",
        "ecFEV1": "ecFEV1",
        "uecFEV1": "uecFEV1",
        "O2Sat": "O2 Saturation",
        "HO2Sat": "Healthy O2 Saturation",
        "ecFEF25-75%ecFEV1": "ecFEF2575%ecFEV1",
        "AR": "AR",
        "IA": "IA",
        "DE": "Days Elapsed",
    }


def abbr_to_colname(name: str):
    # elapsed=False
    # if "_day" in name:
    #     elapsed = True
    #     name, day_n = str.split(name, "_day")
    #     day_n = int(day_n)
    colname = abbr_to_colname_dict().get(name, "Invalid abbreviation")
    if colname == "Invalid abbreviation":
        raise ValueError(f"Invalid abbreviation: {name}")
    # if elapsed:
    #     colname = f"{colname} {day_n} days"

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


class DiscreteVariableNode:
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
        # In reality it is an interval, but using bin_width to keep the same naming convention as VariableNode
        self.bin_width = interval
        self.values = np.arange(a, b + interval, interval)
        self.card = len(self.values)


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
        self.card = len(self.midbins)
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

    def sample_from_bin(self, bin, n=1):
        """
        Sample a set of possible values from a bin
        """
        rng = np.random.default_rng()
        return rng.uniform(bin[0], bin[1], n)

    def sample(self, n=1, p=None):
        """
        Randomly select a midbins from the variable prior's distribution
        If the variable was continuous but has been discretised, it returns a random value inside the sampled bin range
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

        def sample_from_midbin(midbin, n=1):
            """
            When using a continuous variable discretised with bins, we sample from the bin
            TODO: use sample from bin function
            """
            rng = np.random.default_rng()
            return rng.uniform(
                midbin - self.bin_width / 2, midbin + self.bin_width / 2, n
            )

        # Otherwise sample from the bins
        sample = np.array(list(map(sample_from_midbin, midbins)))
        # Make sure to return a 1d array
        return sample.reshape(-1)

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
        assert (
            len(p) == self.card
        ), f"Prior must have the var's cardinality ({self.card}), got {len(p)}"
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

        (lower_val, upper_val) = self.get_bins_arr()[idx]
        return f"[{lower_val:.2f}; {upper_val:.2f})", idx

    def get_point_message(self, obs_val):
        # Create an array with 1 at the index of the evidence and 0 elsewhere
        message = np.zeros(self.card)
        idx = self.get_bin_for_value(obs_val)[1]
        message[idx] = 1
        return message

    def get_val_at_quantile(self, p, q):
        """
        Returns the value at a given quantile q given input probability distribution p
        Assuming uniform distribution within each bin
        """
        # Compute the cumulative distribution function
        cdf = np.cumsum(p)
        # Find the index of the first value that is greater than q
        idx = np.where(cdf >= q, cdf, np.inf).argmin()
        # Get corresponding bin values
        val_l, val_r = self.get_bins_arr()[idx]
        p_l = 0 if idx == 0 else cdf[idx - 1]
        p_r = cdf[idx]

        def get_interpolated_value(val_l, cdf_l, val_r, cdf_r, q):
            return val_l + (val_r - val_l) * (q - cdf_l) / (cdf_r - cdf_l)

        return get_interpolated_value(val_l, p_l, val_r, p_r, q)

    def get_IPR(self, p, p1=0.15865, p2=0.84135):
        """
        Returns the distribution's interpercentile range (inclusive) given an array of probabilities

        IQR: q1=0.25, q3=0.75
        1sigma = 68.27%, p1=(100-68.27)/2=15.865%, p2=100-15.865=84.135%
        2sigma = 95.45%, p1=(100-95.45)/2=2.275%, p2=100-2.275=97.725%
        3sigma = 99.73%, p1=(100-99.73)/2=0.135%, p2=100-0.135=99.865%
        """
        return self.get_val_at_quantile(p, p2) - self.get_val_at_quantile(p, p1)

    def bin_up(self, arr, normalise=False):
        """
        Bin up the input array into the variable's bins
        Return a probability distribution
        """
        hist, _ = np.histogram(arr, bins=self.get_bins_for_hist())
        # Normalize the histogram if hist is not empty
        if normalise and sum(hist) > 0:
            hist = hist / sum(hist)
        return hist

    def get_bins_for_hist(self):
        return np.append(self.bins, self.b)


class SharedVariableNode(VariableNode):
    """
    In longitudinal models, a shared variable is a variable that is
    connected to a factor belonging to a time plate with a lower time resolution.
    The shared variable is able to store messages coming from that factor across
    time (ex: days), and operate on them to be used as virtual messages in the
    slicing belief propagation algorithm.
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
        """
        Save the message for a given day into the variable's dictionary of messages
        """
        assert new_message.shape == (
            self.card,
        ), "The message must have the same shape as the variable's cardinality"
        # Always replace the message for that day, even if it already exists
        self.vmessages[day_key] = new_message

    def set_agg_virtual_message(self, vmessage, new_message):
        """
        The new aggregated virtual message is the multiplication of all messages coming in
        from the factor nodes to the shared variable node.

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


class TemporalVariableNode(VariableNode):
    """
    In longitudinal models, a temporal variable is a variable that is
    connected to a factor belonging to a different time plate on the right
    and on the left (if applicable).
    The temporal variable is able to store messages coming from that factor,
    and operate on them to be used as virtual messages in the slicing belief
    propagation algorithm.
    """

    def __init__(self, name: str, a, b, bin_width):
        # The prior must be uniform since the virtual message supersede it.
        super().__init__(name, a, b, bin_width, {"type": "uniform"})
        self.vmessages = {}
        # The first day prior has to be set as a virtual message
        self.first_day_prior = None
        # For each new day, the virtual message is the previous/next days'
        # posteriors multiplied by the change factor, from both sides
        self.change_cpt = None

    def set_first_day_prior(self, prior_dict):
        """
        The prior must be of the same card as the var's card
        """
        self.first_day_prior = self.set_prior(prior_dict)

    def set_change_cpt(self, change_cpt):
        """
        2 first dimensions must be the same as the variable's cardinality
        """
        assert (
            len(change_cpt.shape) == 3
        ), f"CPT must have 3 dimensions, got {len(change_cpt.shape)}"
        assert (
            change_cpt.shape[0] == self.card
        ), f"CPT first dimension must have the var's cardinality ({self.card}), got {change_cpt.shape[0]}"
        assert (
            change_cpt.shape[1] == self.card
        ), f"CPT second dimension must have the var's cardinality ({self.card}), got {change_cpt.shape[1]}"
        self.change_cpt = change_cpt

    def add_or_update_posterior(self, date_key, new_message):
        """
        Saves the posterior for a given day
        The day key is a string in the format "%Y-%m-%d"
        """
        assert new_message.shape == (
            self.card,
        ), "The message must have the same shape as the variable's cardinality"
        # Always replace the message for that day, even if it already exists
        self.vmessages[date_key] = new_message

    def reset(self):
        self.vmessages = {}

    def calc_days_elapsed(self, date1, date2):
        assert date1 < date2, "Days order 'date1 < date2' not respected"
        days_elapsed = (date2 - date1).days
        if days_elapsed > self.change_cpt.shape[2]:
            raise ValueError(
                f"Can't process {days_elapsed} days (date1 = {date1}, date2 = {date2})"
            )
        return (date2 - date1).days

    def get_virtual_message(self, curr_date, prev_date=None, next_date=None):
        """
        The virtual message for the temporal variable on this day is influenced by its directly neighbouring days,
        through the change factor (change_cpt).
        The neigbouring days might not be related to consecutive dates in the calendar.
        The virtual message can be seen as acting as a prior for the variable on this day. That is why a temporal
        variable's prior is set to uniform by the constructor.

        Dates are datetime objects
        """
        prev_day_key = prev_date.strftime("%Y-%m-%d") if prev_date is not None else None
        next_day_key = next_date.strftime("%Y-%m-%d") if next_date is not None else None

        # Contribution from the previous day
        if prev_date is None:
            # On day 1, the prior is the first_day_prior
            prev_day_m = self.first_day_prior
        else:
            # The previous day's posterior updated through the change factor acts as the current days's prior.
            prev_day_posterior = self.vmessages.get(prev_day_key)
            # Assert that the message exists
            assert (
                prev_day_posterior is not None
            ), f"Posterior for pre day {prev_date} is missing"

            # Compute factor node message
            de_idx = self.calc_days_elapsed(prev_date, curr_date) - 1
            cpt_for_de = self.change_cpt[:, :, de_idx]
            prev_day_m = np.matmul(cpt_for_de, prev_day_posterior)
            prev_day_m *= prev_day_m / prev_day_m.sum()

        # Contribution from the next day
        if next_date is None:
            # Return uniform message
            next_day_m = np.ones(self.card) / self.card
        else:
            # The next day's posterior also propagates a belief to the current day's variable, through the change factor.
            next_day_posterior = self.vmessages.get(next_day_key)
            assert (
                next_day_posterior is not None
            ), f"Posterior for next day {next_date} is missing"

            # Compute factor node message
            de_idx = self.calc_days_elapsed(curr_date, next_date) - 1
            cpt_for_de = self.change_cpt[:, :, de_idx]
            next_day_m = np.matmul(next_day_posterior, cpt_for_de)

        vmessage = prev_day_m * next_day_m
        return vmessage / vmessage.sum()


class CutsetConditionedTemporalVariableNode(VariableNode):
    """
    In longitudinal models, a temporal variable is a variable that is
    connected to a factor belonging to a different time plate on the right
    and on the left (if applicable).
    The temporal variable is able to store messages coming from that factor,
    and operate on them to be used as virtual messages in the slicing belief
    propagation algorithm.

    When the graph is resolved with cutset conditioning, it will be
    conditioned with different states. Hence, a temporal variable has to
    duplicate itself for each state of the cutset conditioning.

    The only difference is that the virtual messages dict becomes a list of dicts,
    the length of the list equals the number of states that the variable can be conditionned upon.
    """

    def __init__(self, name: str, a, b, bin_width, n_states):
        # The prior must be uniform since the virtual message supersede it.
        super().__init__(name, a, b, bin_width, {"type": "uniform"})
        self.n_states = n_states
        # There is on set of vmessages per state
        self.vmessages = [{} for _ in range(n_states)]
        # The first day prior has to be set as a virtual message
        # Is common to all states
        self.first_day_prior = None
        # For each new day, the virtual message is the previous/next days'
        # posteriors multiplied by the change factor, from both sides
        # Is common to all states
        self.change_cpt = None

    def set_first_day_prior(self, prior_dict):
        """
        The prior must be of the same card as the var's card
        """
        self.first_day_prior = self.set_prior(prior_dict)

    def set_change_cpt(self, change_cpt):
        """
        2 first dimensions must be the same as the variable's cardinality
        """
        assert (
            len(change_cpt.shape) == 3
        ), f"CPT must have 3 dimensions, got {len(change_cpt.shape)}"
        assert (
            change_cpt.shape[0] == self.card
        ), f"CPT first dimension must have the var's cardinality ({self.card}), got {change_cpt.shape[0]}"
        assert (
            change_cpt.shape[1] == self.card
        ), f"CPT second dimension must have the var's cardinality ({self.card}), got {change_cpt.shape[1]}"
        self.change_cpt = change_cpt

    def add_or_update_posterior(self, state_n, date_key, new_message, debug=False):
        """
        Saves the posterior for a given day
        The day key is a string in the format "%Y-%m-%d"
        """
        assert new_message.shape == (
            self.card,
        ), "The message must have the same shape as the variable's cardinality"
        # Always replace the message for that day, even if it already exists
        self.vmessages[state_n][date_key] = new_message
        if debug:
            print(
                f"CutsetCondTemporalVarNode: Posterior for {self.name} on {date_key}, state {state_n} has been updated to {new_message}"
            )

    def reset(self):
        self.vmessages = {}

    def calc_days_elapsed(self, date1, date2):
        assert date1 < date2, "Days order 'date1 < date2' not respected"
        days_elapsed = (date2 - date1).days
        if days_elapsed > self.change_cpt.shape[2]:
            raise ValueError(
                f"Can't process {days_elapsed} days (date1 = {date1}, date2 = {date2})"
            )
        return (date2 - date1).days

    def get_virtual_message(
        self, state_n, curr_date, prev_date=None, next_date=None, debug=False
    ):
        """
        The virtual message for the temporal variable on this day is influenced by its directly neighbouring days,
        through the change factor (change_cpt).
        The neigbouring days might not be related to consecutive dates in the calendar.
        The virtual message can be seen as acting as a prior for the variable on this day. That is why a temporal
        variable's prior is set to uniform by the constructor.

        Dates are datetime objects
        """
        prev_day_key = prev_date.strftime("%Y-%m-%d") if prev_date is not None else None
        next_day_key = next_date.strftime("%Y-%m-%d") if next_date is not None else None
        print(
            f"Get virtual message for {self.name} on {curr_date} with cutset cond. state {state_n}, wrt to prev {prev_day_key} and next {next_day_key}"
        )

        def calc_days_elapsed(date1, date2):
            assert date1 < date2, "Days order 'date1 < date2' not respected"
            days_elapsed = (date2 - date1).days
            if days_elapsed > self.change_cpt.shape[2]:
                raise ValueError(
                    f"Can't process {days_elapsed} days (date1 = {date1}, date2 = {date2})"
                )
            return (date2 - date1).days

        # Contribution from the previous day
        if prev_date is None:
            # On day 1, the prior is the first_day_prior
            prev_day_m = self.first_day_prior
            if debug:
                print(f"No prev day, using first day prior: {prev_day_m}")
        else:
            # The previous day's posterior updated through the change factor acts as the current days's prior.
            prev_day_posterior = self.vmessages[state_n].get(prev_day_key)
            # Assert that the message exists
            assert (
                prev_day_posterior is not None
            ), f"Posterior for pre day {prev_date} is missing"

            # Compute factor node message
            de_idx = calc_days_elapsed(prev_date, curr_date) - 1
            cpt_for_de = self.change_cpt[:, :, de_idx]
            prev_day_m = np.matmul(cpt_for_de, prev_day_posterior)
            prev_day_m = prev_day_m / prev_day_m.sum()
            if debug:
                print(f"Prev day posterior: {prev_day_posterior}")
                print(f"Prev day message: {prev_day_m}")

        # Contribution from the next day
        if next_date is None:
            # Return uniform message
            next_day_m = np.ones(self.card) / self.card
            if debug:
                print("No next day, using uniform message")
        else:
            # The next day's posterior also propagates a belief to the current day's variable, through the change factor.
            next_day_posterior = self.vmessages[state_n].get(next_day_key)
            assert (
                next_day_posterior is not None
            ), f"Posterior for next day {next_date} is missing"

            # Compute factor node message
            de_idx = calc_days_elapsed(curr_date, next_date) - 1
            cpt_for_de = self.change_cpt[:, :, de_idx]
            next_day_m = np.matmul(next_day_posterior, cpt_for_de)
            if debug:
                print(f"Next day posterior: {next_day_posterior}")
                print(f"Next day message: {next_day_m}")

        vmessage = prev_day_m * next_day_m
        vmessage = vmessage / vmessage.sum()
        if debug:
            print(f"Returning virtual message: {vmessage}")
        return vmessage


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
    nbinsX = X.card
    nbinsY = Y.card
    nbinsZ = Z.card
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
