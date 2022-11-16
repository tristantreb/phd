import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from partition import *


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
