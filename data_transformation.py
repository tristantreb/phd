import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


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


# Compute FEV1 desaturation, i.e. the threshold of FEV1 at which the O2 starts to desaturate
def desaturation_FEV1_for_variable(O2_FEV1, x_var, n_var_groups, predicted=False):
    """
    Partition input data in several O2 and <input variable> groups.
    Computes the avg FEV1 for each pair of group.
    Creates a figure superposing avg_lung_function_var(<input variable>) line plots for each O2 groups
    :param O2_FEV1: input dataframe with all columns
    :param predicted: x variable is FEV1 if False, else uses FEV1 % Predicted
    :param lung_function_var: name of the <x_var data>
    :param n_var_groups: # groups to partition the <input variable> data
    :return: dataframe with (O2 group, <input variables> group, avg FEV1)
    """

    lung_function_var = 'FEV1 % Predicted' if predicted else 'FEV1'

    # create groups
    # define O2 thresholds
    o2_thresholds = [93, 96, 98, 99]
    O2_FEV1['O2 Group'], o2_labels = partition_given_thresholds(O2_FEV1['O2 Saturation'], o2_thresholds, True)
    O2_FEV1[x_var + ' Group'], var_labels = partition_in_n_equal_groups(O2_FEV1[x_var], n_var_groups, True)

    # create dataframe to host avg fev1 wrt each pari of group
    desaturation_FEV1 = pd.DataFrame(columns=['O2 Group', x_var + ' Group', 'Avg ' + lung_function_var, 'Datapoints'])

    # Compute averaged FEV1 for each pair of different n_o2_groups, n_var_group
    for o2_group in o2_labels:
        for n_var_group in var_labels:
            mask = (O2_FEV1['O2 Group'] == o2_group) & (O2_FEV1[x_var + ' Group'] == n_var_group)
            datapoints = sum(mask)
            avg_lung_function_var = O2_FEV1[mask][lung_function_var].mean()
            desaturation_FEV1.loc[len(desaturation_FEV1.index)] = [o2_group, n_var_group, avg_lung_function_var,
                                                                   datapoints]

    # order data by categories
    desaturation_FEV1['O2 Group'] = pd.Categorical(desaturation_FEV1['O2 Group'],
                                                   categories=o2_labels,
                                                   ordered=True)
    desaturation_FEV1[x_var + ' Group'] = pd.Categorical(desaturation_FEV1[x_var + ' Group'],
                                                         categories=var_labels,
                                                         ordered=True)
    desaturation_FEV1.sort_values(['O2 Group', x_var + ' Group'], inplace=True)

    fig_with_legend = px.scatter(desaturation_FEV1, x=x_var + ' Group', y='Avg ' + lung_function_var, color='O2 Group',
                                 size='Datapoints',
                                 title='Desaturation FEV1 for {} {} groups'.format(n_var_groups, x_var))
    fig_with_legend.update_xaxes(categoryorder='array', categoryarray=var_labels)

    fig_temp = px.line(desaturation_FEV1, x=x_var + ' Group', y='Avg ' + lung_function_var, color='O2 Group')

    fig_with_lines = go.Figure(data=fig_with_legend.data + fig_temp.data)

    return fig_with_legend, fig_with_lines, desaturation_FEV1
