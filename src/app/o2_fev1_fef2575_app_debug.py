import os

import dash_bootstrap_components as dbc
from dash import Dash, dcc, html

import app.assets.styles as s
import app.components.sliders as sliders
from app.callbacks.one_callback_app import (
    build_fev1_fef2575_o2sat_with_factor_graph_debug,
)
from app.components.clinical_profile_input import clinical_profile_input_layout
from app.components.inf_settings import var_to_infer_select_layout
from app.components.observed_vars_checklist import (
    fev1_fef2575_o2sat_observed_vars_checklist_layout,
)
from app.components.inf_settings import priors_settings_layout

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
            "What is your lung health?",
            style={"textAlign": "center", "padding-top": "0px"},
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
        var_to_infer_select_layout,
        priors_settings_layout,
        html.Div(
            "3. Select observed values, and view your the state of your lungs health:",
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
                "margin-left": s.width_px(0.24),
                "padding-top": "0px",
            },
        ),
        html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(sliders.fev1_slider_layout),
                        dbc.Col(sliders.fef2575_slider_layout),
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

build_fev1_fef2575_o2sat_with_factor_graph_debug(app)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8054)
