import dash_bootstrap_components as dbc
from dash import dcc

import app.assets.styles as s
import app.components.sliders as sliders

healthy_vars_row = dbc.Row(
    [
        dbc.Col(
            [
                dcc.Loading(
                    children=[
                        dcc.Graph(
                            id="HFEV1-dist",
                            style={"display": "none"},
                        )
                    ],
                ),
                sliders.hfev1_slider_layout,
            ],
            width=3,
        ),
        dbc.Col(
            [""],
            width=3,
        ),
        dbc.Col(
            [
                dcc.Loading(
                    children=[
                        dcc.Graph(
                            id="HO2Sat-dist",
                            style={"display": "none"},
                        )
                    ],
                ),
                sliders.ho2sat_slider_layout,
            ],
            width=3,
        ),
    ],
    style={
        "font-size": s.font_size(),
        "padding-top": "20px",
        "padding-bottom": "0px",
    },
)

ar_row = dbc.Row(
    [
        dbc.Col(
            [""],
            width=3,
        ),
        dbc.Col(
            [
                dcc.Loading(
                    children=[
                        dcc.Graph(
                            id="AR-dist",
                            style={"display": "none"},
                        )
                    ],
                ),
                sliders.AR_slider_layout,
            ],
            width=3,
        ),
        dbc.Col(
            [""],
            width=3,
        ),
    ],
    style={
        "font-size": s.font_size(),
        "padding-top": "20px",
        "padding-bottom": "0px",
    },
)

ia_row = dbc.Row(
    [
        dbc.Col(
            [""],
            width=3,
        ),
        dbc.Col(
            [
                dcc.Loading(
                    children=[
                        dcc.Graph(
                            id="IA-dist",
                        )
                    ],
                ),
            ],
            width=3,
        ),
        dbc.Col(
            [""],
            width=3,
        ),
    ],
    style={
        "font-size": s.font_size(),
        "padding-top": "20px",
        "padding-bottom": "0px",
    },
)

o2satffa_row = dbc.Row(
    [
        dbc.Col(
            [""],
            width=3,
        ),
        dbc.Col(
            [""],
            width=3,
        ),
        dbc.Col(
            [
                dcc.Loading(
                    children=[
                        dcc.Graph(
                            id="O2SatFFA-dist",
                        )
                    ],
                ),
            ],
            width=3,
        ),
    ],
    style={
        "font-size": s.font_size(),
        "padding-top": "20px",
        "padding-bottom": "0px",
    },
)

u_o2sat_fev1_row = dbc.Row(
    [
        dbc.Col(
            [
                dcc.Loading(
                    children=[
                        dcc.Graph(
                            id="uFEV1-dist",
                        )
                    ],
                ),
            ],
            width=3,
        ),
        dbc.Col([""], width=3),
        dbc.Col(
            [
                dcc.Loading(
                    children=[
                        dcc.Graph(
                            id="uO2Sat-dist",
                        )
                    ],
                ),
            ],
            width=3,
        ),
    ],
    style={
        "font-size": s.font_size(),
        "padding-top": "20px",
        "padding-bottom": "0px",
    },
)
