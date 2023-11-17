# Lunch app with "python interactive_inference_healthy_gauss.py"
import os

import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from plotly.subplots import make_subplots

import src.inference.helpers as ih
import src.modelling_fev1.pred_fev1 as pred_fev1
import src.models.builders as mb
import src.models.helpers as mh

"""
Solving: "Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized" error
https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial
"""
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

app = Dash(__name__, external_stylesheets=[dbc.themes.UNITED])

# TODO
# - dbc input type number does return None if not number or outside authorised range
# - dbc input type number returns a string, not a number (have to convert it)

app.layout = dbc.Container(
    [
        html.H2("Lung Health's Diagnostic Tool", style={"textAlign": "center"}),
        html.Div(
            [
                html.H4("Individual's clinical profile:"),
                dbc.InputGroup(
                    [
                        dbc.InputGroupText("Sex"),
                        dbc.Select(
                            id="sex-select",
                            options=[
                                {"label": "Male", "value": "Male"},
                                {"label": "Female", "value": "Female"},
                            ],
                            value="Male",
                        ),
                    ]
                ),
                dbc.InputGroup(
                    [
                        dbc.InputGroupText("Age (years)"),
                        dbc.Input(
                            type="number",
                            min=0,
                            max=100,
                            value=30,
                            id="age-input",
                            debounce=True,
                        ),
                    ]
                ),
                dbc.InputGroup(
                    [
                        dbc.InputGroupText("Height (cm)"),
                        dbc.Input(
                            type="number",
                            min=0,
                            max=300,
                            value=175,
                            id="height-input",
                            debounce=True,
                        ),
                    ]
                ),
            ],
            style={"width": "300px"},
        ),
        dcc.Loading(
            id="graph-loader", type="default", children=[dcc.Graph(id="lung-graph")]
        ),
        html.Div(
            dbc.Form(
                [
                    dbc.Label("FEV1 observed:"),
                    dcc.Slider(
                        id="FEV1-slider",
                        min=0,
                        max=6,
                        step=0.1,
                        value=3,
                        marks={
                            1: "1 L",
                            2: "2 L",
                            3: "3 L",
                            4: "4 L",
                            5: "5 L",
                        },
                        tooltip={"always_visible": True, "placement": "bottom"},
                    ),
                ],
                style={"margin-left": "90px", "margin-right": "900px"},
            ),
        ),
    ],
    fluid=True,
)


@app.callback(
    Output("lung-graph", "figure"),
    Output("FEV1-slider", "min"),
    Output("FEV1-slider", "max"),
    # Variables
    Input("sex-select", "value"),
    Input("age-input", "value"),
    Input("height-input", "value"),
    # Evidences
    Input("FEV1-slider", "value"),
)
def update_inference(sex: str, age: int, height: int, FEV1_obs: float):
    # MODEL
    print(
        f"Model inputs: sex: {sex}, age: {age}, height: {height}, FEV1_obs: {FEV1_obs}"
    )
    # TODO: why not int by default?
    height = int(height)
    age = int(age)
    healthy_FEV1_prior = {"type": "default", "height": height, "age": age, "sex": sex}
    (
        model,
        inf_alg,
        FEV1,
        HFEV1,
        AB,
    ) = mb.build_HFEV1_AB_FEV1(healthy_FEV1_prior)

    # INFERENCE
    print("Inference user input: FEV1 set to", FEV1_obs)

    res_u = ih.infer(inf_alg, [HFEV1], [[FEV1, FEV1_obs]])
    res_ab = ih.infer(inf_alg, [AB], [[FEV1, FEV1_obs]])

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

    fig.add_trace(go.Bar(y=HFEV1.prior[:,0], x=HFEV1.bins), row=1, col=1)
    fig["data"][0]["marker"]["color"] = "blue"
    fig["layout"]["xaxis"]["title"] = "Prior for " + HFEV1.name

    fig.add_trace(go.Bar(y=AB.prior[:,0], x=AB.bins), row=1, col=3)
    fig["data"][1]["marker"]["color"] = "green"
    fig["layout"]["xaxis2"]["title"] = "Prior for " + AB.name

    fig.add_trace(go.Bar(y=res_u.values, x=HFEV1.bins), row=2, col=1)
    fig["data"][2]["marker"]["color"] = "blue"
    fig["layout"]["xaxis3"]["title"] = HFEV1.name

    fig.add_trace(go.Bar(y=res_ab.values, x=AB.bins), row=2, col=3)
    fig["data"][3]["marker"]["color"] = "green"
    fig["layout"]["xaxis4"]["title"] = AB.name

    fig.update_layout(showlegend=False, height=600, width=1200)

    return fig, FEV1.a, FEV1.b


app.run_server(debug=True, port=8051, use_reloader=False)
