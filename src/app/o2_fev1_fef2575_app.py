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

"""
Solving: "Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized" error
https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial
"""
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

app = Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE, "./assets/styles.css"])

app.layout = dbc.Container(
    [
        html.Div(
            [
                "Tristan Trébaol, Floto's lab, Cambridge University",
                html.Br(),
                "Contact: tpbt2[at]cam.ac.uk",
                html.Br(),
                html.Strong("This page is a work in progress"),
            ],
            style={"font-size": "12px", "text-align": "right", "padding-right": "0px"},
        ),
        html.H2(
            "What is your lung's health?",
            style={"textAlign": "center", "padding-top": "0px"},
        ),
        html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(clinical_profile_input_layout),
                        dbc.Col(
                            dcc.Checklist(
                                id="observed-vars-checklist",
                                options={
                                    "FEF25-75": "FEF25-75",
                                    "FEV1": "FEV1",
                                    "O2 saturation": "O2 saturation",
                                },
                                value=["FEV1", "O2 saturation"],
                            )
                        ),
                    ]
                )
            ]
        ),
        # clinical_profile_input_layout,
        html.Div(
            "2. Select your FEV1, FEF25-75, and O2 saturation, and analyse your lung's health variables:",
            style={
                "textAlign": "left",
                "padding-top": "20px",
                "padding-bottom": "10px",
            },
        ),
        dcc.Loading(
            id="graph-loader",
            type="default",
            children=[
                dcc.Graph(id="lung-graph"),
            ],
        ),
        html.Div(
            id="FEF25-75-prct-FEV1-output",
            style={
                "font-size": s.font_size(),
                "margin-left": s.width_px(2 / 6),
                "padding-top": "0px",
            },
        ),
        html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(sliders.fev1_slider_layout),
                        dbc.Col(
                            [
                                dcc.Graph(
                                    id="FEF25-75-dist", style={"display": "none"}
                                ),
                                sliders.fef2575_slider_layout,
                            ]
                        ),
                        dbc.Col(sliders.O2Sat_slider_layout),
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
    Output("FEF25-75-dist", "style"),
    Output("FEF25-75-slider-container", "style"),
    Input("observed-vars-checklist", "value"),
)
def show_slider_or_graph_FEF2575(
    observed_vars_checklist: List[str],
):
    if "FEF25-75" in observed_vars_checklist:
        return {"display": "none"}, {"display": "block"}
    return {"display": "block"}, {"display": "none"}


build_fev1_fef2575_o2sat_with_factor_graph(app)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
