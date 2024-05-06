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
    style={"margin-left": "0px", "margin-right": "0px"},
)

fef2575_slider_layout = dbc.Form(
    [
        dbc.Label("FEF25-75 observed:"),
        dcc.Slider(
            id="FEF25-75-slider",
            min=0,
            max=6,
            step=0.1,
            value=3,
            marks={
                1: "1 L/s",
                2: "2 L/s",
                3: "3 L/s",
                4: "4 L/s",
                5: "5 L/s",
            },
            tooltip={
                "always_visible": True,
                "placement": "bottom",
            },
        ),
    ],
    style={"margin-left": "0px", "margin-right": "0px"},
)

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
    style={"margin-left": "0px", "margin-right": "0px"},
)
