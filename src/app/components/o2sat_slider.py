import dash_bootstrap_components as dbc
from dash import dcc

O2Sat_slider_layout = dbc.Form(
    [
        dbc.Label("O2 saturation observed:"),
        dcc.Slider(
            id="O2Sat-slider",
            min=80,
            max=100,
            step=1,
            value=98,
            tooltip={
                "always_visible": True,
                "placement": "bottom",
            },
        ),
    ],
    style={"margin-left": "10", "margin-right": "200px"},
)
