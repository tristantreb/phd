import numpy as np


# Function to extract one column from the data
# TODO: check that there's only one measurement per day
def extract_measure(measurements_in, label):
    # Could also filter by Recording Type
    measurements_out = measurements_in[measurements_in[label].notnull()][['ID', 'Date recorded', label]]
    print("{} contains {} measurements".format(label, measurements_out.shape[0]))
    return measurements_out


# Functions to partition the data in n balanced groups

# Balance the population density by partitioning the data 3-fold equally
def partition_in_n_equal_groups(series, n, labels=False):
    """
    Split a Panda Series into n equal groups
    :param series: Panda Series of input data
    :param n: number of groups
    :param labels: bool to return group labels (or not)
    :return: series where values is replaced with group label
    """
    q_quantiles = np.linspace(0, 1, n + 1)[1:-1]
    q_values = series.quantile(q_quantiles).round(decimals=1).to_list()
    series = series.apply(lambda x: _value_to_group(x, q_values))

    if labels:
        return series, _values_to_group_labels(q_values)
    else:
        return series


def partition_given_thresholds(series, thresholds, labels=False):
    """
    Partition a Panda Series into groups according to a given set of thresholds :param series: Panda Series of input
    data :param thresholds: list of floats corresponding to the wanted interval thresholds. Don't include the
    boundaries, e.g. 40.0 returns groups for: <40.0, >= 40.0 :param labels: bool to return group labels (or not)
    :return: series where values is replaced with group label
    """
    series = series.apply(lambda x: _value_to_group(x, thresholds))

    if labels:
        return series, _values_to_group_labels(thresholds)
    else:
        return series


def _value_to_group(value, thresholds):
    """
    Map group to value :param value: :param thresholds: list of floats corresponding to the wanted interval
    thresholds. Don't include the boundaries, e.g. 40.0 returns groups for: <40.0, >= 40.0 :return:
    """
    for i in range(0, len(thresholds)):
        if thresholds[i] == thresholds[0]:
            if value < thresholds[i]:
                return '<' + str(thresholds[i])
        else:
            if thresholds[i - 1] <= value < thresholds[i]:
                return '[' + str(thresholds[i - 1]) + ';' + str(thresholds[i]) + '['

        if thresholds[i] == thresholds[-1]:
            if value >= thresholds[i]:
                return '>=' + str(thresholds[i])


def _values_to_group_labels(values):
    """
    Create group labels corresponding to values intervals
    :param values: list of values
    :return: list of group labels ordered from lowest to highest values intervals
    """
    group_labels = ['<' + str(values[0])]
    for i in range(0, len(values) - 1):
        group_labels.append('[' + str(values[i]) + ';' + str(values[i + 1]) + '[')
    group_labels.append('>=' + str(values[-1]))
    return group_labels