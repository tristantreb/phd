import dash_bootstrap_components as dbc
from dash import html

import src.app.assets.styles as styles
from src.models.helpers import name_to_abbr_dict

font_size = styles.font_size()
select_style = {"font-size": font_size}
input_group_text_style = {"width": "150px", "font-size": font_size}

# Remove ecfev1 and o2sat from list
name_to_abbr_dict = name_to_abbr_dict()
name_to_abbr_dict.pop("ecFEV1 (L)")
name_to_abbr_dict.pop("O2 saturation (%)")
var_to_infer_options = list(
    map(lambda name: {"label": name, "value": name}, name_to_abbr_dict.keys())
)

inference_settings_layout = html.Div(
    [
        html.Div(
            "2. Select the inference settings:",
            style={"textAlign": "left", "padding-top": "0px", "padding-bottom": "5px"},
        ),
        dbc.InputGroup(
            [
                dbc.InputGroupText("Variable to infer", style=input_group_text_style),
                dbc.Select(
                    id="var-to-infer-select",
                    options=var_to_infer_options,
                    value="Healthy FEV1 (L)",
                    style=select_style,
                ),
            ],
        ),
        dbc.InputGroup(
            [
                dbc.InputGroupText(
                    "Inactive alveoli prior", style=input_group_text_style
                ),
                dbc.Select(
                    id="ia-prior-select",
                    options=[
                        {"label": "uniform", "value": "uniform"},
                        {"label": "from Breathe data", "value": "breathe"},
                    ],
                    value="breathe",
                    style=select_style,
                ),
            ],
        ),
    ],
    style={"width": "430px"},
)
