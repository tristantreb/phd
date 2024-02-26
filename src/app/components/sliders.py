import dash_bootstrap_components as dbc
from dash import dcc
import src.app.assets.styles as s

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
    style={"margin-left": "0px", "margin-right": s.width_px(1/6)},
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
    style={"margin-left": s.width_px(1/6), "margin-right": "0px"},
)

# o2_fev1_sliders_layout = html.Div(
#             [
#                 dbc.Row(
#                     [
#                         dbc.Col(fev1_slider_layout),
#                         dbc.Col(O2Sat_slider_layout),
#                     ],
#                     style={"font-size": s.font_size(), "padding-top": "20px", "padding-bottom": "0px"},
#                 ),
#             ]
#         ),