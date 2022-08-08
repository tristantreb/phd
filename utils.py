import numpy as np
import random as rand
from scipy.stats import bernoulli


def bernoulliSample(p, valueTrue, valueFalse):
    """
    :param p: probability
    :param valueTrue: label for true
    :param valueFalse: label for false
    :return: random variate following a bernouilli distribution
    """
    output = {
        0: valueFalse,
        1: valueTrue,
    }
    return output.get(bernoulli.rvs(p))


def threeStatesSample(p, values):
    """
    input: p and values as 3 elements array
    output: sampled value depending on probability distribution
    """
    randVal = rand.random()
    if randVal < p[0]:
        return values[0]
    elif p[0] <= randVal < (p[0] + p[1]):
        return values[1]
    else:
        return values[2]


def nStatesSample(p, values):
    """
    input: p and values as n elements array
    output: sampled value depending on probability distribution
    """
    randVal = rand.random()
    p_sum = 0
    for idx in range(len(p)):
        if p_sum <= randVal < p_sum + p[idx]: return values[idx]
        p_sum += p[idx]


def gaussianSample(moments=[0, 1], nDecimals=2):
    """
    moments[0]: mean
    moments[1]: standard deviation (by def the second moment is variance)
    """
    return round(np.random.normal(loc=moments[0], scale=moments[1]), nDecimals)


def normalise(input_dict):
    """
    :param input_dict: a one layer dictionary
    :return: dictionary with normalised values
    """
    values = np.array(list(input_dict.values()))
    return dict(zip(input_dict.keys(), values / sum(values)))


def get_second_level_keys(cpt_B_A, init_value=0):
    """
    returns a dict with second level keys and values set to init_value
    """
    return {key: init_value for key in cpt_B_A[next(iter(cpt_B_A))]}


def get_first_level_keys(cpt_B_A, init_value=0):
    """
    returns a dict with first level keys and values set to init_value
    """
    return {key: init_value for key in cpt_B_A.keys()}
