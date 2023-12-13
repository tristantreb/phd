# Lunch app with "python interactive_inference_healthy_gauss.py"
import os

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from plotly.subplots import make_subplots

import src.inference.helpers as ih
import src.modelling_o2.helpers as o2h
import src.models.helpers as mh
import src.models.o2_fev1_point_in_time as model
import src.modelling_o2.o2sat as o2sat

"""
Solving: "Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized" error
https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial
"""
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

app = Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE])
# app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
# app = Dash(__name__, external_stylesheets=[dbc.themes.MORPH])

# TODO
# - dbc input type number does return None if not number or outside authorised range
# - dbc input type number returns a string, not a number (have to convert it)

app.layout = dbc.Container(
    [
        html.H2("My lung's health", style={"textAlign": "center"}),
        html.Div(
            [
                # html.H4("Individual's clinical profile:"),
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
                dcc.Store(id="HO2Sat", storage_type="session"),
                dcc.Store(id="O2SatFFA", storage_type="session"),
            ],
        ),
        html.Div(
            [
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
                dbc.Form(
                    [
                        dbc.Label("O2 saturation observed:"),
                        dcc.Slider(
                            id="O2Sat-slider",
                            min=80,
                            max=100,
                            step=1,
                            value=98,
                            # marks={
                            #     1: "1 L",
                            #     2: "2 L",
                            #     3: "3 L",
                            #     4: "4 L",
                            #     5: "5 L",
                            # },
                            tooltip={"always_visible": True, "placement": "bottom"},
                        ),
                    ],
                    style={"margin-left": "900px", "margin-right": "90px"},
                ),
            ]
        ),
    ],
    fluid=True,
)


@app.callback(
    Output("HFEV1", "data"),
    Output("FEV1", "data"),
    Output("AR", "data"),
    Output("HO2Sat", "data"),
    Output("O2SatFFA", "data"),
    # Inputs
    Input("sex-select", "value"),
    Input("age-input", "value"),
    Input("height-input", "value"),
)
def calc_cpts(sex: str, age: int, height: int):
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
        HO2Sat,
        O2SatFFA,
    ) = model.calc_cpts(hfev1_prior, ho2sat_prior)

    # Encode variables
    HFEV1 = mh.encode_node_variable(HFEV1)
    FEV1 = mh.encode_node_variable(FEV1)
    AR = mh.encode_node_variable(AR)
    HO2Sat = mh.encode_node_variable(HO2Sat)
    O2SatFFA = mh.encode_node_variable(O2SatFFA)

    return HFEV1, FEV1, AR, HO2Sat, O2SatFFA


@app.callback(
    Output("lung-graph", "figure"),
    Output("FEV1-slider", "min"),
    Output("FEV1-slider", "max"),
    # Variables
    Input("HFEV1", "data"),
    Input("FEV1", "data"),
    Input("AR", "data"),
    Input("HO2Sat", "data"),
    Input("O2SatFFA", "data"),
    # Evidences
    Input("FEV1-slider", "value"),
    Input("O2Sat-slider", "value"),
)
def model_and_inference(
    HFEV1, ecFEV1, AR, HO2Sat, O2SatFFA, FEV1_obs: float, O2Sat_obs: float
):
    """
    Decodes inputs from JSON format, build model, runs inference, and returns a figure
    """
    # Decode input vars
    HFEV1 = mh.decode_node_variable(HFEV1)
    ecFEV1 = mh.decode_node_variable(ecFEV1)
    AR = mh.decode_node_variable(AR)
    HO2Sat = mh.decode_node_variable(HO2Sat)
    O2SatFFA = mh.decode_node_variable(O2SatFFA)

    O2Sat = o2sat.add_gaussian_noise(O2Sat_obs, HO2Sat.bin_width)

    # Build model
    _, inf_alg = model.build_pgmpy_model(HFEV1, ecFEV1, AR, HO2Sat, O2SatFFA)

    # INFERENCE
    print("Inference user input: FEV1 =", FEV1_obs, ", O2Sat =", O2Sat_obs)

    res_hfev1 = ih.infer(inf_alg, [HFEV1], [[ecFEV1, FEV1_obs]])
    res_ar = ih.infer(inf_alg, [AR], [[ecFEV1, FEV1_obs]])
    res_ho2sat = ih.infer(inf_alg, [HO2Sat], [[ecFEV1, FEV1_obs]])
    res_o2satffa = ih.infer(inf_alg, [O2SatFFA], [[ecFEV1, FEV1_obs]])

    # PLOT
    # Priors take 1x1 cells, posteriors take 2x2 cells
    prior = {"type": "bar", "colspan": 2}
    posterior = {"type": "bar", "rowspan": 2, "colspan": 2}

    viz_layout = [
        [prior, None, None, None, prior, None],
        [posterior, None, None, None, posterior, None],
        [None, None, None, None, None, None],
        [None, None, prior, None, None, None],
        [None, None, posterior, None, None, None],
        [None, None, None, None, None, None],
        [None, None, None, None, posterior, None],
        [None, None, None, None, None, None],
        [None, None, None, None, None, None],
        [None, None, None, None, posterior, None],
        [None, None, None, None, None, None],
    ]

    fig = make_subplots(
        rows=np.shape(viz_layout)[0], cols=np.shape(viz_layout)[1], specs=viz_layout
    )

    fev1_min = ecFEV1.a
    fev1_max = ecFEV1.b
    o2sat_min = 80
    o2sat_max = 100

    # HFEV1
    ih.plot_histogram(fig, HFEV1, HFEV1.prior[:, 0], fev1_min, fev1_max, 1, 1, False)
    fig["data"][0]["marker"]["color"] = "black"

    ih.plot_histogram(fig, HFEV1, res_hfev1.values, fev1_min, fev1_max, 2, 1)
    fig["data"][1]["marker"]["color"] = "black"

    # HO2Sat
    ih.plot_histogram(
        fig, HO2Sat, HO2Sat.prior[:, 0], o2sat_min, o2sat_max, 1, 5, False
    )
    fig["data"][2]["marker"]["color"] = "blue"
    o2h.add_o2sat_normal_range_line(fig, max(HO2Sat.prior[:, 0]), 1, 5)

    ih.plot_histogram(fig, HO2Sat, res_ho2sat.values, o2sat_min, o2sat_max, 2, 5)
    fig["data"][3]["marker"]["color"] = "blue"
    o2h.add_o2sat_normal_range_line(fig, max(res_ho2sat.values), 2, 5)

    # AR
    ih.plot_histogram(fig, AR, AR.prior[:, 0], AR.a, AR.b, 4, 3, False)
    fig["data"][4]["marker"]["color"] = "green"

    ih.plot_histogram(fig, AR, res_ar.values, AR.a, AR.b, 5, 3)
    fig["data"][5]["marker"]["color"] = "green"

    # O2SatFFA
    ih.plot_histogram(fig, O2SatFFA, res_o2satffa.values, o2sat_min, o2sat_max, 7, 5)
    fig["data"][6]["marker"]["color"] = "cyan"
    o2h.add_o2sat_normal_range_line(fig, max(res_o2satffa.values), 7, 5)

    # O2Sat
    ih.plot_histogram(fig, O2Sat, O2Sat.prior[:, 0], o2sat_min, o2sat_max, 10, 5)
    o2h.add_o2sat_normal_range_line(fig, max(O2Sat.prior[:, 0]), 10, 5)

    fig.update_layout(
        showlegend=False, height=800, width=1400, font=dict(size=10), bargap=0.01
    )
    fig.update_traces(marker_line_width=0)

    return fig, ecFEV1.a, ecFEV1.b


app.run_server(debug=True, port=8051, use_reloader=False)
