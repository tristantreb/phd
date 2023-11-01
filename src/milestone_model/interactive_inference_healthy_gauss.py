# Lunch app with "python interactive_inference_healthy_gauss.py"
import os
import sys

import dash_bootstrap_components as dbc
import model_helpers as mh
import model_lung_health
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from plotly.subplots import make_subplots

sys.path.append("../")
import data.biology as bio

"""
Solving: "Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized" error
https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial
"""
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

app = Dash(__name__, external_stylesheets=[dbc.themes.UNITED])

app.layout = html.Div(
    [
        html.H2(
            "Interactive inference on the lung's health model",
            style={"textAlign": "center"},
        ),
        html.Br(),
        html.Div(
            f"Individual clinical profile:"
        ),
        html.Div(
            [
                html.P("Sex"),
                dbc.Select(
                    id="sex-select",
                    options=[
                        {"label": "Male", "value": "Male"},
                        {"label": "Female", "value": "Female"},
                    ],
                    value="Male"
                ),
                html.P("Age (years))"),
                dbc.Input(type="number, min=0, max=100", value=30, id="age-input"),
                html.P("Height (cm)"),
                dbc.Input(type="number, min=0, max=300", value=175, id="height-input"),
            ],
        ),
        dcc.Graph(id="lung-graph"),
        html.Div(
            [
                dcc.Slider(
                    id="FEV1-slider",
                    min=0,
                    max=6,
                    # min=FEV1.bins[0],
                    # max=FEV1.bins[-2],
                    value=3,
                    # marks={0: "0.2", (len(AB.bins) - 1): "5.9"},
                )
            ],
            style={
                "transform": "scale(1.2)",
                "margin-left": "90px",
                "margin-right": "900px",
            },
        ),
        dcc.Store(id="model", storage_type="memory"),
        dcc.Store(id="FEV1-node", storage_type="memory"),
        dcc.Store(id="HFEV1-node", storage_type="memory"),
        dcc.Store(id="prior-HFEV1", storage_type="memory"),
        dcc.Store(id="AB-node", storage_type="memory"),
        dcc.Store(id="prior-AB", storage_type="memory"),
    ]
)

# @app.callback(
#     Output("FEV1-slider"),
#     [Input("FEV1", )]
# )

@app.callback(
    Output("model", "data"),
    Output("FEV1-node", "data"),
    Output("HFEV1-node", "data"),
    Output("prior-HFEV1", "data"),
    Output("AB-node", "data"),
    Output("prior-AB", "data"),
    Input("sex-select", "value"),
    Input("age-input", "value"),
    Input("height-input", "value"),
)
def update_model(sex: str, age: int, height: int):
    pred_FEV1, pred_FEV1_std = list(
        map(bio.calc_predicted_fev1(height, age, sex).get, ["Predicted FEV1", "std"])
    )
    # healthy_FEV1_prior={"type":"uniform"}
    healthy_FEV1_prior = {"type": "gaussian", "mu": pred_FEV1, "sigma": pred_FEV1_std}
    (
        model,
        FEV1,
        HFEV1,
        prior_HFEV1,
        AB,
        prior_AB,
    ) = model_lung_health.build_HFEV1_AB_FEV1(healthy_FEV1_prior)
    return model, FEV1, HFEV1, prior_HFEV1, AB, prior_AB

@app.callback(
    Output("lung-graph", "figure"),
    # Variables
    Input("model", "data"),
    Input("HFEV1-node", "data"),
    Input("prior-HFEV1", "data"),
    Input("AB-node", "data"),
    Input("prior-AB", "data"),
    Input("FEV1-node", "data"),
    # Evidences
    Input("FEV1-slider", "value"),
)
def update_inference(model, HFEV1, prior_HFEV1, FEV1, AB, prior_AB, FEV1_obs: float):
    print("user input: FEV1 set to", FEV1_obs)

    [_fev1_bin, fev1_idx] = mh.get_bin_for_value(FEV1_obs, FEV1.bins)

    res_u = model_lung_health.infer(model, [HFEV1], [[FEV1, FEV1_obs]])
    res_ab = model_lung_health.infer(model, [AB], [[FEV1, FEV1_obs]])

    n_var_rows = 1
    prior = {"type": "bar"}
    posterior = {"type": "bar", "rowspan": 2, "colspan": 2}

    fig = make_subplots(
        # Prior takes a 1x1 cell, Posterior takes a 2x2 cell
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

    fig.add_trace(go.Bar(y=prior_HFEV1.values, x=HFEV1.bins[:-1]), row=1, col=1)
    fig["data"][0]["marker"]["color"] = "blue"
    fig["layout"]["xaxis"]["title"] = "Prior for " + HFEV1.name

    fig.add_trace(go.Bar(y=prior_AB.values, x=AB.bins[:-1]), row=1, col=3)
    fig["data"][1]["marker"]["color"] = "green"
    fig["layout"]["xaxis2"]["title"] = "Prior for " + AB.name

    fig.add_trace(go.Bar(y=res_u.values, x=HFEV1.bins[:-1]), row=2, col=1)
    fig["data"][2]["marker"]["color"] = "blue"
    fig["layout"]["xaxis3"]["title"] = HFEV1.name

    fig.add_trace(go.Bar(y=res_ab.values, x=AB.bins[:-1]), row=2, col=3)
    fig["data"][3]["marker"]["color"] = "green"
    fig["layout"]["xaxis4"]["title"] = AB.name

    fig.update_layout(showlegend=False, height=600, width=1200)

    # Add text box with FEV1 value
    fig.add_annotation(
        x=0,
        y=-0.2,
        text=f"FEV1 = {FEV1_obs:.2f} L",
        showarrow=False,
        font=dict(size=16),
        xref="paper",
        yref="paper",
    )
    return fig


app.run_server(debug=True, port=8051, use_reloader=False)
