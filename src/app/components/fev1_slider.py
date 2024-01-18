import dash_bootstrap_components as dbc
from dash import dcc

fev1_slider_layout = dbc.Form(
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
            tooltip={
                "always_visible": True,
                "placement": "bottom",
            },
        ),
    ],
    style={"margin-left": "10px", "margin-right": "200px"},
)
