import os

import dash_bootstrap_components as dbc
from dash import Dash, dcc, html

from src.app.callbacks.build_model_and_infer import model_and_inference_callback
from src.app.callbacks.build_variables import build_variables_callback
from src.app.components.fev1_slider import fev1_slider_layout
from src.app.components.id_info import id_info_layout
from src.app.components.o2sat_slider import O2Sat_slider_layout

"""
Solving: "Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized" error
https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial
"""
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

app = Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE, "./assets/styles.css"])
server = app.server

app.layout = dbc.Container(
    [
        html.H2("My lung's health", style={"textAlign": "center"}),
        id_info_layout,
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


build_variables_callback(app)
model_and_inference_callback(app)

if __name__ == "__main__": app.run(debug=False, host='0.0.0.0', port=8050)