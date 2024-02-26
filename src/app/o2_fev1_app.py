import os

import dash_bootstrap_components as dbc
from dash import Dash, dcc, html

from src.app.callbacks.one_callback_app import build_all
from src.app.components.clinical_profile_input import clinical_profile_input_layout
import src.app.components.sliders as sliders
import src.app.assets.styles as s

"""
Solving: "Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized" error
https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial
"""
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

app = Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE, "./assets/styles.css"])
server = app.server

app.layout = dbc.Container(
    [
        html.Div(
            ["Tristan Trébaol, Floto's lab, Cambridge University", html.Br(), "Contact: tpbt2[at]cam.ac.uk", html.Br(), html.Strong("This page is a work in progress")],
            style={"font-size": "12px", "text-align": "right", "padding-right": "0px"},
        ),
        html.H2("What is your lung's health?", style={"textAlign": "center", "padding-top": "0px"}),
        clinical_profile_input_layout,
        html.Div(
        "2. Select your FEV1 and O2 saturation, and analyse your lung's health variables:",
        style={"textAlign": "left", "padding-top": "20px", "padding-bottom": "10px"},
        ),
        dcc.Loading(
            id="graph-loader",
            type="default",
            children=[
                dcc.Graph(id="lung-graph"),
            ]
        ),
        html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(sliders.fev1_slider_layout),
                        dbc.Col(sliders.O2Sat_slider_layout),
                    ],
                    style={"font-size": s.font_size(), "padding-top": "20px", "padding-bottom": "0px"},
                ),
            ]
        ),
    ],
    style={"padding-left": "20px", "padding-right": "10px"},
    fluid=True,
)

build_all(app)

if __name__ == "__main__": app.run(debug=True, host='0.0.0.0', port=8050)