import os
from typing import List

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, dcc, html

import src.app.assets.styles as s
import src.app.components.sliders as sliders
from src.app.callbacks.one_callback_app import (
    build_fev1_fef2575_o2sat_with_factor_graph,
)
from src.app.components.clinical_profile_input import clinical_profile_input_layout
from src.app.components.observed_vars_checklist import observed_vars_checklist_layout

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
                        dbc.Col(observed_vars_checklist_layout),
                    ]
                )
            ]
        ),
        # clinical_profile_input_layout,
        html.Div(
            "3. Select observed values, and view your the state of your lungs health:",
            style={
                "textAlign": "left",
                "padding-top": "20px",
                "padding-bottom": "10px",
            },
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


@app.callback(
    Output("FEV1-dist", "style"),
    Output("FEV1-slider-container", "style"),
    Output("O2-saturation-dist", "style"),
    Output("O2-saturation-slider-container", "style"),
    Output("FEF25-75-dist", "style"),
    Output("FEF25-75-slider-container", "style"),
    Input("observed-vars-checklist", "value"),
)
def show_slider_or_graph_FEF2575(
    observed_vars_checklist: List[str],
):

    if "FEF25-75" in observed_vars_checklist:
        fef2575_style = {"display": "none"}, {"display": "block"}
    else:
        fef2575_style = {"display": "block"}, {"display": "none"}
    if "O2 saturation" in observed_vars_checklist:
        o2sat_style = {"display": "none"}, {"display": "block"}
    else:
        o2sat_style = {"display": "block"}, {"display": "none"}
    if "FEV1" in observed_vars_checklist:
        fev1_style = {"display": "none"}, {"display": "block"}
    else:
        fev1_style = {"display": "block"}, {"display": "none"}
    return *fev1_style, *o2sat_style, *fef2575_style


build_fev1_fef2575_o2sat_with_factor_graph(app)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
