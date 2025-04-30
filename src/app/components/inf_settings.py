import dash_bootstrap_components as dbc
from dash import html

import app.assets.styles as styles
from models.helpers import name_to_abbr_dict

font_size = styles.font_size("M")
select_style = {"font-size": font_size}
input_group_text_style = {"width": "150px", "font-size": font_size}

# Remove ecfev1 and o2sat from list
name_to_abbr_dict = name_to_abbr_dict()
name_to_abbr_dict.pop("ecFEV1 (L)")
name_to_abbr_dict.pop("O2 saturation (%)")
var_to_infer_options = list(
    map(lambda name: {"label": name, "value": name}, name_to_abbr_dict.keys())
)

var_to_infer_select_layout = html.Div(
    [
        html.Div(
            "Select the variable to infer:",
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
    ],
    style={"width": "430px"},
)

priors_settings_layout = html.Div(
    [
        html.Div(
            "Select the priors:",
            style={"textAlign": "left", "padding-top": "0px", "padding-bottom": "5px"},
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
                    value="uniform",
                    style=select_style,
                ),
            ],
        ),
        dbc.InputGroup(
            [
                dbc.InputGroupText("AR prior", style=input_group_text_style),
                dbc.Select(
                    id="ar-prior-select",
                    options=[
                        {"label": "uniform", "value": "uniform"},
                        {
                            "label": "uniform in log space",
                            "value": "uniform in log space",
                        },
                        {
                            "label": "uniform message to HFEV1",
                            "value": "uniform message to HFEV1",
                        },
                        {
                            "label": "breathe (2 days model, ecFEV1, ecFEF25-75)",
                            "value": "breathe (2 days model, ecFEV1, ecFEF25-75)",
                        },
                        {
                            "label": "breathe (1 day model, O2Sat, ecFEV1)",
                            "value": "breathe (1 day model, O2Sat, ecFEV1)",
                        },
                        {
                            "label": "breathe (2 days model, ecFEV1, ecFEF25-75, add mult noise)",
                            "value": "breathe (2 days model, ecFEV1, ecFEF25-75, add mult noise)",
                        },
                    ],
                    value="uniform",
                    style=select_style,
                ),
            ],
        ),
    ],
    style={"width": "430px"},
)
