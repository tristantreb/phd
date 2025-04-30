import os

import dash_bootstrap_components as dbc
from dash import Dash, dcc, html

import app.assets.styles as s
import app.components.sliders as sliders
from app.callbacks.one_callback_app import build_fev1_fef2575_o2sat_with_factor_graph
from app.callbacks.show_slider_or_graph import (
    show_slider_or_graph_for_observed_measures,
)
from app.components.clinical_profile_input import clinical_profile_input_layout
from app.components.inf_settings import priors_settings_layout
from app.components.observed_vars_checklist import (
    fev1_fef2575_o2sat_observed_vars_checklist_layout,
)

"""
Solving: "Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized" error
https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial
"""
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

app = Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE, "./assets/styles.css"])
server = app.server
app.title = "My lungs health"

app.layout = dbc.Container(
    [
        html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            html.H2(
                                "What is your lungs health?",
                                style={
                                    "textAlign": "center",
                                    "padding-left": "100px",
                                },
                            )
                        ),
                        dbc.Col(
                            html.Div(
                                [
                                    "Tristan Trébaol, Floto's lab, Cambridge University",
                                    html.Br(),
                                    "Contact: tpbt2[at]cam.ac.uk",
                                    html.Br(),
                                    html.Strong("This page is a work in progress"),
                                ],
                                style={
                                    "font-size": s.font_size("S"),
                                    "text-align": "right",
                                    "padding-right": "0px",
                                },
                            ),
                            width=3,
                        ),
                    ]
                )
            ],
            style={"padding-top": "20px", "padding-bottom": "20px"},
        ),
        html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(clinical_profile_input_layout, width=3),
                        dbc.Col(fev1_fef2575_o2sat_observed_vars_checklist_layout),
                    ]
                )
            ]
        ),
        priors_settings_layout,
        html.Div(
            "Select observed values, and view your the state of your lungs health:",
            style={
                "textAlign": "left",
                "padding-top": "20px",
                "padding-bottom": "10px",
            },
        ),
        html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Loading(
                                    children=[
                                        dcc.Graph(
                                            id="HFEV1-dist",
                                            style={"display": "none"},
                                        )
                                    ],
                                ),
                                sliders.hfev1_slider_layout,
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [""],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                dcc.Loading(
                                    children=[
                                        dcc.Graph(
                                            id="HO2Sat-dist",
                                            style={"display": "none"},
                                        )
                                    ],
                                ),
                                sliders.ho2sat_slider_layout,
                            ],
                            width=3,
                        ),
                    ],
                    style={
                        "font-size": s.font_size(),
                        "padding-top": "20px",
                        "padding-bottom": "0px",
                    },
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [""],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                dcc.Loading(
                                    children=[
                                        dcc.Graph(
                                            id="AR-dist",
                                            style={"display": "none"},
                                        )
                                    ],
                                ),
                                sliders.AR_slider_layout,
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [""],
                            width=3,
                        ),
                    ],
                    style={
                        "font-size": s.font_size(),
                        "padding-top": "20px",
                        "padding-bottom": "0px",
                    },
                ),
            ]
        ),
        dcc.Loading(children=[dcc.Graph(id="lung-graph")]),
        html.Div(
            id="FEF25-75-prct-FEV1-output",
            style={
                "font-size": s.font_size(),
                "margin-left": s.width_px(0.24),
                "padding-top": "0px",
            },
        ),
        html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Loading(
                                    children=[
                                        dcc.Graph(
                                            id="FEV1-dist",
                                            style={"display": "none"},
                                        )
                                    ],
                                ),
                                sliders.fev1_slider_layout,
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                dcc.Loading(
                                    children=[
                                        dcc.Graph(
                                            id="FEF25-75-dist",
                                            style={"display": "none"},
                                        )
                                    ],
                                ),
                                sliders.fef2575_slider_layout,
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                dcc.Loading(
                                    children=[
                                        dcc.Graph(
                                            id="O2-saturation-dist",
                                            style={"display": "none"},
                                        )
                                    ],
                                ),
                                sliders.O2Sat_slider_layout,
                            ],
                            width=3,
                        ),
                    ],
                    style={
                        "font-size": s.font_size(),
                        "padding-top": "20px",
                        "padding-bottom": "0px",
                    },
                ),
            ]
        ),
    ],
    style={"padding-left": "20px", "padding-right": "10px"},
    fluid=True,
)


build_fev1_fef2575_o2sat_with_factor_graph(app)
show_slider_or_graph_for_observed_measures(app)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8051)
