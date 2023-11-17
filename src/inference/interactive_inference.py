import os

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from plotly.subplots import make_subplots

import src.modelling_fev1.pred_fev1 as pred_fev1
import src.models.builders as lung_health_models
import src.models.helpers as mh
import src.inference.helpers as ih

"""
Solving: "Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized" error
https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial
"""
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

app = Dash(__name__)


set_age = 26
set_height = 175
set_sex = "Male"
pred_FEV1, pred_FEV1_std = pred_fev1.calc_predicted_FEV1_linear(
    set_height, set_age, set_sex
)

# healthy_FEV1_prior={"type":"uniform"}
healthy_FEV1_prior = {"type": "gaussian", "mu": pred_FEV1, "sigma": pred_FEV1_std}
inference, FEV1, U, prior_u, C, prior_c = lung_health_models.build(healthy_FEV1_prior)

app.layout = html.Div(
    [
        html.H2("Interactive inference on the lung's health model"),
        html.Div(
            f"Individual with predicted FEV1 of {pred_FEV1:.2f}L (std {pred_FEV1_std} L) given height {set_height} cm, age {set_age} years, {set_sex}"
        ),
        dcc.Graph(id="lung-graph"),
        html.P("FEV1:"),
        dcc.Slider(
            id="fev1",
            min=FEV1.a,
            max=FEV1.b,
            value=3,
            marks={0: "0.2", (len(C.bins)): "5.9"},
        ),
        # dcc.Dropdown(['Gaussian', 'Uniform'], 'Uniform', id='healthy-fev1-prior'),
        # html.Div(id='output-container')
    ]
)


@app.callback(
    Output("lung-graph", "figure"),
    Input("fev1", "value"),
    # Input('healthy-fev1-prior', 'value')
)
def display(fev1: float):
    print("user input: FEV1 set to", fev1)

    [_fev1_bin, fev1_idx] = ih.get_bin_for_value(fev1, FEV1)

    res_u = inference.query(
        variables=[U.name], evidence={FEV1.name: fev1_idx}, show_progress=False
    )
    res_c = inference.query(
        variables=[C.name], evidence={FEV1.name: fev1_idx}, show_progress=False
    )

    n_var_rows = 1
    prior = {"type": "bar"}
    posterior = {"type": "bar", "rowspan": 2, "colspan": 2}

    fig = make_subplots(
        # Prior takes 1 row, Posterior takes 2 rows
        rows=n_var_rows * 3,
        cols=5,
        specs=[
            # priors
            [prior, None, prior, None, None],
            # posteriors
            [posterior, None, posterior, None, None],
            [None, None, None, None, None],
        ],
    )

    fig.add_trace(go.Bar(y=prior_u.values, x=U.bins), row=1, col=1)
    fig["data"][0]["marker"]["color"] = "blue"
    fig["layout"]["xaxis"]["title"] = "Prior for " + U.name

    fig.add_trace(go.Bar(y=prior_c.values, x=C.bins), row=1, col=3)
    fig["data"][1]["marker"]["color"] = "green"
    fig["layout"]["xaxis2"]["title"] = "Prior for " + C.name

    fig.add_trace(go.Bar(y=res_u.values, x=U.bins), row=2, col=1)
    fig["data"][2]["marker"]["color"] = "blue"
    fig["layout"]["xaxis3"]["title"] = U.name

    fig.add_trace(go.Bar(y=res_c.values, x=C.bins), row=2, col=3)
    fig["data"][3]["marker"]["color"] = "green"
    fig["layout"]["xaxis4"]["title"] = C.name

    fig.update_layout(showlegend=False, height=600, width=1200)
    return fig


app.run_server(debug=True, port=8049, use_reloader=False)
