from math import ceil  # math.ceil converts to int, np.ceil returns float

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Define color for measurements done in exacerbation period
def get_ex_color(opacity=0.7):
    return "rgba(213,094,000,{})".format(opacity)


# Define color for measurements done in stable period
def get_stable_color(opacity=0.7):
    return "rgba(000,114,178,{})".format(opacity)


# Create O2-FEV scatter plot with displots on x and y axes
# This is the final plot of the O2-FEV analysis
def plot_o2_fev_with_displots(O2_FEV1, x, y, ex_column):
    # Set colors and point opacities
    opacity_scatter = 0.6
    opacity_distplot = 0.7
    ex_color_scatter = get_ex_color(opacity_scatter)
    stable_color_scatter = get_stable_color(opacity_scatter)
    ex_color_distplot = get_ex_color(opacity_distplot)
    stable_color_distplot = get_stable_color(opacity_distplot)

    # Create a figure with 3 subplots with shared x and y axes and different row/column sizes.
    fig = make_subplots(
        rows=2,
        cols=2,
        shared_xaxes=True,
        shared_yaxes=True,
        column_widths=[0.8, 0.2],
        row_heights=[0.3, 0.7],
        vertical_spacing=0.02,
        horizontal_spacing=0.005,
    )

    # Separate x and y data by exacerbation label
    x_exacerbated = O2_FEV1[x][O2_FEV1[ex_column] == True]
    x_stable = O2_FEV1[x][O2_FEV1[ex_column] == False]
    y_exacerbated = O2_FEV1[y][O2_FEV1[ex_column] == True]
    y_stable = O2_FEV1[y][O2_FEV1[ex_column] == False]

    # Add scatter plot
    fig.add_trace(
        go.Scatter(
            x=x_stable,
            y=y_stable,
            mode="markers",
            # name="Stable",
            marker=dict(
                size=5,
                color=stable_color_scatter,
                line=dict(width=0.2, color="DarkSlateGrey"),
            ),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_exacerbated,
            y=y_exacerbated,
            mode="markers",
            # name="Exacerbated",
            marker=dict(
                size=5,
                color=ex_color_scatter,
                line=dict(width=0.2, color="DarkSlateGrey"),
            ),
        ),
        row=2,
        col=1,
    )
    # Define the number of bins for the distribution plots
    bins_n_stable = 40
    # Bin width should be 1/15 of the span of the stable data
    bins_width = (max(x_stable) - min(x_stable)) / bins_n_stable
    # Bins number for the exacerabted data
    bins_n_ex = ceil((max(x_exacerbated) - min(x_exacerbated)) / bins_width)

    # Add displot for x
    fig.add_trace(
        go.Histogram(
            x=x_stable,
            histnorm="probability",
            # name="Stable",
            nbinsx=bins_n_stable,
            marker=dict(color=stable_color_distplot),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=x_exacerbated,
            histnorm="probability",
            nbinsx=bins_n_ex,
            # name="Exacerbated",
            marker=dict(color=ex_color_distplot),
        ),
        row=1,
        col=1,
    )
    # Add displot for y
    fig.add_trace(
        go.Histogram(
            y=y_stable,
            histnorm="probability",
            # Number of bins automatically set is good enough because O2 saturation is a dicsrete variable with a small values span
            # nbinsy=nbins,
            name="Stable ({} points)".format(len(x_stable)),
            marker=dict(color=stable_color_distplot),
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Histogram(
            y=y_exacerbated,
            histnorm="probability",
            # nbinsy=nbins,
            name="Exacerbated ({} points)".format(len(x_exacerbated)),
            marker=dict(color=ex_color_distplot),
        ),
        row=2,
        col=2,
    )
    fig.update_layout(barmode="overlay")
    # Set x axis title to x, and range to min max of O2_FEV1[x]
    fig.update_xaxes(title_text=x, row=2, col=1)
    # Set y axis title to y
    fig.update_yaxes(title_text=y, row=2, col=1)
    # Remove duplicated legends
    for i in range(0, len(fig.data) - 2):
        fig.data[i].showlegend = False
    # Update fig size
    fig.update_layout(height=600, width=1400)

    return fig


# Set the name of x=O2 and y=FEV variables depending on the choosen options (smoothed, normalised, etc.)
def set_x_y_vars(
    with_predicted_labels,
    with_predicted_fev1,
    is_smoothed_fev,
    is_smoothed_o2,
    is_normalised,
):
    predictive_classifier = "from APE pred" if with_predicted_labels else ""
    smoothed_fev = " smoothed" if is_smoothed_fev else ""
    smoothed_o2 = " smoothed" if is_smoothed_o2 else ""
    predicted = " % Predicted" if with_predicted_fev1 else ""
    normalised = " norm" if is_normalised else ""

    ex_column = "Is Exacerbated" if with_predicted_labels else "Exacerbation Labels"
    prefix = "{}{}".format(predictive_classifier, normalised)
    # y = 'O2 Saturation{}'.format(smoothed)
    y = "O2 Saturation{}{}".format(smoothed_o2, normalised)
    x = "FEV1{}{}{}".format(predicted, smoothed_fev, normalised)
    return prefix, ex_column, x, y


# Create a scatter plot with px with subsampled x data and y
def plot_subsampled_scatter(
    x,
    y,
    O2_FEV1,
    ex_column="Is Exacerbated",
    ex_color="rgba(213,094,000,0.5)",
    stable_color="rgba(000,114,178,0.5)",
    random_state=42,
):
    # Subsample
    ex = O2_FEV1[O2_FEV1[ex_column] == True]
    non_ex = O2_FEV1[O2_FEV1[ex_column] == False]
    subsample_factor = round(ex.shape[0])
    O2_FEV1_sub = pd.concat(
        [ex, non_ex.sample(subsample_factor, random_state=random_state)], axis=0
    )

    fig = px.scatter(
        O2_FEV1_sub,
        x=x,
        y=y,
        color=ex_column,
        color_discrete_map={True: ex_color, False: stable_color},
    )
    fig.update_layout(
        height=400,
        width=1150,
        title="Subsampled measurement done in stable period".format(x, y),
    )
    return fig


# Raw O2-FEV1 scatter plot for a given patient ID
def plot_o2_fev1_raw_for_id(
    O2_FEV1,
    id,
    plotsdir,
    x="FEV1 % Predicted",
    time_scale=False,
    show=False,
    save=False,
):
    y = "O2 Saturation"
    if time_scale:
        fig = px.scatter(
            O2_FEV1[O2_FEV1.ID == id],
            y="O2 Saturation",
            x=x,
            color="Months since study start",
        )
    else:
        fig = px.scatter(O2_FEV1[O2_FEV1.ID == id], y="O2 Saturation", x=x)
    fig.update_xaxes(title_text=x)
    fig.update_layout(width=600, height=400)

    if show:
        fig.show()
    if save:
        fig.write_image(
            "{}/Patient raw plots/{} {}-{} raw.pdf".format(plotsdir, id, x, y)
        )
