import dash_bootstrap_components as dbc
from dash import html

import app.assets.styles as styles

clinical_profile_box_style = {"width": "200px", "font-size": styles.font_size("L")}
input_group_text_style = {"width": "105px", "font-size": styles.font_size("M")}
select_style = {"font-size": styles.font_size("M")}

clinical_profile_input_layout = html.Div(
    [
        html.Div(
            "Enter your clinical profile:",
            style={"textAlign": "left", "padding-top": "0px", "padding-bottom": "5px"},
        ),
        dbc.InputGroup(
            [
                dbc.InputGroupText("Sex", style=input_group_text_style),
                dbc.Select(
                    id="sex-select",
                    options=[
                        {"label": "Male", "value": "Male"},
                        {"label": "Female", "value": "Female"},
                    ],
                    value="Male",
                    style=select_style,
                ),
            ],
        ),
        dbc.InputGroup(
            [
                dbc.InputGroupText("Age (years)", style=input_group_text_style),
                dbc.Input(
                    type="number",
                    min=18,
                    max=100,
                    value=30,
                    step=1,
                    id="age-input",
                    debounce=True,
                    style=select_style,
                ),
            ],
            id="styled-numeric-input1",
        ),
        html.Div(id="age-error-message", style={"color": "#dc3545"}),
        dbc.InputGroup(
            [
                dbc.InputGroupText("Height (cm)", style=input_group_text_style),
                dbc.Input(
                    type="number",
                    min=140,
                    max=220,
                    value=175,
                    step=1,
                    id="height-input",
                    debounce=True,
                    style=select_style,
                ),
            ],
            id="styled-numeric-input2",
        ),
    ],
    style=clinical_profile_box_style,
)
