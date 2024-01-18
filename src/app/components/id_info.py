import dash_bootstrap_components as dbc
from dash import html

id_info_layout = html.Div(
    [
        dbc.InputGroup(
            [
                dbc.InputGroupText("Sex"),
                dbc.Select(
                    id="sex-select",
                    options=[
                        {"label": "Male", "value": "Male"},
                        {"label": "Female", "value": "Female"},
                    ],
                    value="Male",
                ),
            ]
        ),
        dbc.InputGroup(
            [
                dbc.InputGroupText("Age (years)"),
                dbc.Input(
                    type="number",
                    min=18,
                    max=100,
                    value=30,
                    step=1,
                    id="age-input",
                    debounce=True,
                ),
            ],
            id="styled-numeric-input",
        ),
        html.Div(id="age-error-message", style={"color": "#dc3545"}),
        dbc.InputGroup(
            [
                dbc.InputGroupText("Height (cm)"),
                dbc.Input(
                    type="number",
                    min=140,
                    max=220,
                    value=175,
                    step=1,
                    id="height-input",
                    debounce=True,
                ),
            ],
            id="styled-numeric-input",
        ),
    ],
    style={"width": "300px"},
)
