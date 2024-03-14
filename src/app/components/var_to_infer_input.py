import dash_bootstrap_components as dbc
from dash import html

import src.app.assets.styles as styles
from src.models.helpers import name_to_abbr_dict

font_size = styles.font_size()
select_style = {"font-size": font_size}

# Remove ecfev1 and o2sat from list
name_to_abbr_dict = name_to_abbr_dict()
name_to_abbr_dict.pop("ecFEV1 (L)")
name_to_abbr_dict.pop("O2 saturation (%)")
select_options = list(
    map(lambda name: {"label": name, "value": name}, name_to_abbr_dict.keys())
)

var_to_infer_input_layout = html.Div(
    [
        html.Div(
            "2. Select the variable to infer:",
            style={"textAlign": "left", "padding-top": "0px", "padding-bottom": "5px"},
        ),
        dbc.InputGroup(
            [
                dbc.InputGroupText("Variable to infer", style={"font-size": font_size}),
                dbc.Select(
                    id="var-to-infer-select",
                    options=select_options,
                    value="Healthy FEV1 (L)",
                    style=select_style,
                ),
            ],
        ),
    ],
    style={"width": "500px"},
)
