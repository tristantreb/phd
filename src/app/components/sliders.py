import dash_bootstrap_components as dbc
from dash import dcc

hfev1_slider_layout = dbc.Form(
    id="HFEV1-slider-container",
    children=[
        dbc.Label("HFEV1 observed:"),
        dcc.Slider(
            id="HFEV1-slider",
            min=1,
            max=6,
            step=0.1,
            value=4.5,
            marks={
                1: "1 L",
                2: "2 L",
                3: "3 L",
                4: "4 L",
                5: "5 L",
            },
            tooltip={
                "placement": "bottom",
            },
        ),
    ],
    style={"display": "block"},
)

fev1_slider_layout = dbc.Form(
    id="FEV1-slider-container",
    children=[
        dbc.Label("FEV1 observed:"),
        dcc.Slider(
            id="FEV1-slider",
            min=0.1,
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
                "placement": "bottom",
            },
        ),
    ],
    style={"display": "block"},
)

fef2575_slider_layout = dbc.Form(
    id="FEF25-75-slider-container",
    children=[
        dbc.Label("FEF25-75 observed:"),
        dcc.Slider(
            id="FEF25-75-slider",
            min=0.1,
            max=6,
            step=0.1,
            value=2,
            marks={
                1: "1 L/s",
                2: "2 L/s",
                3: "3 L/s",
                4: "4 L/s",
                5: "5 L/s",
            },
            tooltip={
                "placement": "bottom",
            },
        ),
    ],
    style={"display": "block"},
)

ho2sat_slider_layout = dbc.Form(
    id="HO2Sat-slider-container",
    children=[
        dbc.Label("HO2Sat observed:"),
        dcc.Slider(
            id="HO2Sat-slider",
            min=80,
            max=100,
            step=1,
            value=98,
            marks={
                80: "80%",
                85: "85%",
                90: "90%",
                95: "95%",
                100: "100%",
            },
            tooltip={
                "placement": "bottom",
            },
        ),
    ],
    style={"display": "block"},
)


O2Sat_slider_layout = dbc.Form(
    id="O2-saturation-slider-container",
    children=[
        dbc.Label("O2 saturation observed:"),
        dcc.Slider(
            id="O2Sat-slider",
            min=80,
            max=100,
            step=1,
            value=98,
            marks={
                80: "80%",
                85: "85%",
                90: "90%",
                95: "95%",
                100: "100%",
            },
            tooltip={
                "placement": "bottom",
            },
        ),
    ],
    style={"display": "block"},
)

AR_slider_layout = dbc.Form(
    id="AR-slider-container",
    children=[
        dbc.Label("AR observed:"),
        dcc.Slider(
            id="AR-slider",
            min=0,
            max=90,
            step=2,
            value=30,
            marks={
                0: "0",
                30: "30",
                60: "60",
                90: "90",
            },
            tooltip={
                "placement": "bottom",
            },
        ),
    ],
    style={"display": "block"},
)
