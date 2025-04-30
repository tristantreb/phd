"""
Same app but with an IA prior learnt from the Breathe dataset
"""

import os

import dash_bootstrap_components as dbc
from dash import Dash, dcc, html

from app.callbacks.build_model_and_infer import model_and_inference_callback
from app.callbacks.build_variables import build_variables_with_cf_callback
from app.components.clinical_profile_input import clinical_profile_input_layout
from app.components.sliders import O2Sat_slider_layout, fev1_slider_layout

"""
Solving: "Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized" error
https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial
"""
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

app = Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE, "./assets/styles.css"])

app.layout = dbc.Container(
    [
        html.H2("My lung's health", style={"textAlign": "center"}),
        clinical_profile_input_layout,
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
                dcc.Store(id="IA", storage_type="session"),
                dcc.Store(id="UO2Sat", storage_type="session"),
                dcc.Store(id="O2Sat", storage_type="session"),
            ],
        ),
        html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(fev1_slider_layout),
                        dbc.Col(O2Sat_slider_layout),
                    ]
                ),
            ]
        ),
    ],
    fluid=True,
)


build_variables_with_cf_callback(app)
model_and_inference_callback(app)

app.run_server(debug=True, port=8051, use_reloader=False)
