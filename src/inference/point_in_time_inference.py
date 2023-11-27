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
        html.H2("Lung Health's Tool", style={"textAlign": "center"}),
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
            id="graph-loader",
            type="default",
            children=[
                dcc.Graph(id="lung-graph"),
                dcc.Store(id="HFEV1", storage_type="session"),
                dcc.Store(id="FEV1", storage_type="session"),
                dcc.Store(id="AR", storage_type="session"),
                # dcc.Store(id='HO2Sat', storage_type='session')
                # dcc.Store(id='O2SatFFA', storage_type='session')
            ],
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
    Output("HFEV1", "data"),
    Output("FEV1", "data"),
    Output("AR", "data"),
    # Output("HO2Sat", "data"),
    # Output("O2SatFFA", "data"),
    # Inputs
    Input("sex-select", "value"),
    Input("age-input", "value"),
    Input("height-input", "value"),
)
def compute_cpts(sex: str, age: int, height: int):
    # TODO: why not int by default?
    height = int(height)
    age = int(age)
    hfev1_prior = {"type": "default", "height": height, "age": age, "sex": sex}
    ho2sat_prior = {
        "type": "default",
        "height": height,
        "sex": sex,
    }
    (
        HFEV1,
        FEV1,
        AR,
    ) = mb.calc_point_in_time_cpts(hfev1_prior, ho2sat_prior)

    # Encode variables
    HFEV1 = mh.encode_node_variable(HFEV1)
    FEV1 = mh.encode_node_variable(FEV1)
    # HO2Sat= mh.encode_node_variable(HO2Sat)
    # O2SatFFA= mh.encode_node_variable(O2SatFFA)
    AR = mh.encode_node_variable(AR)

    return HFEV1, FEV1, AR


@app.callback(
    Output("lung-graph", "figure"),
    Output("FEV1-slider", "min"),
    Output("FEV1-slider", "max"),
    # Variables
    Input("HFEV1", "data"),
    Input("FEV1", "data"),
    Input("AR", "data"),
    # Input("HO2Sat", "data"),
    # Input("O2SatFFA", "data"),
    # Evidences
    Input("FEV1-slider", "value"),
)
def model_and_inference(HFEV1, FEV1, AR, FEV1_obs: float):
    """
    Decodes inputs from JSON format, build model, runs inference, and returns a figure
    """
    # Decode input vars
    HFEV1 = mh.decode_node_variable(HFEV1)
    FEV1 = mh.decode_node_variable(FEV1)
    AR = mh.decode_node_variable(AR)
    # HO2Sat = mh.decode_node_variable(HO2Sat)
    # O2SatFFA = mh.decode_node_variable(O2SatFFA)

    # Build model
    _, inf_alg = mb.build_point_in_time_pgmpy_model(HFEV1, FEV1, AR)

    # INFERENCE
    print("Inference user input: FEV1 set to", FEV1_obs)

    res_hfev1 = ih.infer(inf_alg, [HFEV1], [[FEV1, FEV1_obs]])
    res_ar = ih.infer(inf_alg, [AR], [[FEV1, FEV1_obs]])
    # res_ho2sat = ih.infer(inf_alg, [HO2Sat], [[FEV1, FEV1_obs]])
    # res_o2satffa = ih.infer(inf_alg, [O2SatFFA], [[FEV1, FEV1_obs]])

    # PLOT
    n_var_rows = 3
    prior = {"type": "bar"}
    posterior = {"type": "bar", "rowspan": 2, "colspan": 2}

    fig = make_subplots(
        # Prior takes a 1x1 cell, Posterior takes a 2x2 cell
        rows=n_var_rows * 2,
        cols=6,
        specs=[
            [prior, None, None, None, None, None],
            [posterior, None, None, None, None, None],
            [None, None, None, None, None, None],
            [None, None, prior, None, None, None],
            [None, None, posterior, None, None, None],
            [None, None, None, None, None, None],
            # [None, None, None, None, posterior, None],
            # [None, None, None, None, None, None],
            # [None, None, None, None, None, None],
        ],
    )

    # HFEV1
    fig.add_trace(go.Bar(y=HFEV1.prior[:, 0], x=HFEV1.bins), row=1, col=1)
    fig["data"][0]["marker"]["color"] = "black"
    # fig.update_xaxes(title_text="Prior for " + HFEV1.name, row=1, col=1)

    fig.add_trace(go.Bar(y=res_hfev1.values, x=HFEV1.bins), row=2, col=1)
    fig["data"][1]["marker"]["color"] = "black"
    fig.update_xaxes(title_text=HFEV1.name, row=2, col=1)

    # HO2Sat
    # fig.add_trace(go.Bar(y=HO2Sat.prior[:, 0], x=HO2Sat.bins), row=1, col=5)
    # fig["data"][2]["marker"]["color"] = "blue"
    # # fig.update_xaxes(title_text="Prior for " + HO2Sat.name, row=1, col=5)

    # fig.add_trace(go.Bar(y=res_ho2sat.values, x=HO2Sat.bins), row=2, col=5)
    # fig["data"][3]["marker"]["color"] = "blue"
    # fig.update_xaxes(title_text=HO2Sat.name, row=2, col=5)

    # AR
    fig.add_trace(go.Bar(y=AR.prior[:, 0], x=AR.bins), row=4, col=3)
    fig["data"][2]["marker"]["color"] = "green"
    # fig.update_xaxes(title_text="Prior for " + AR.name, row=4, col=3)

    fig.add_trace(go.Bar(y=res_ar.values, x=AR.bins), row=5, col=3)
    fig["data"][3]["marker"]["color"] = "green"
    fig.update_xaxes(title_text=AR.name, row=5, col=3)

    # O2SatFFA
    # fig.add_trace(go.Bar(y=res_o2satffa.values, x=O2SatFFA.bins), row=7, col=5)
    # fig["data"][6]["marker"]["color"] = "cyan"
    # fig.update_xaxes(title_text=O2SatFFA.name, row=7, col=5)

    fig.update_layout(showlegend=False, height=800, width=1400, font=dict(size=8))

    return fig, FEV1.a, FEV1.b


app.run_server(debug=True, port=8051, use_reloader=False)
