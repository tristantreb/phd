from dash import dcc, html

import app.assets.styles as styles

fev1_o2sat_observed_vars_checklist_layout = html.Div(
    [
        html.Div("Select the observed variables:"),
        dcc.Checklist(
            id="observed-vars-checklist",
            options={
                "FEV1": "FEV1",
                "O2 saturation": "O2 saturation",
            },
            value=["FEV1", "O2 saturation"],
        ),
    ],
    style={"font-size": styles.font_size()},
)

fev1_fef2575_o2sat_observed_vars_checklist_layout = html.Div(
    [
        html.Div("Select the observed variables:"),
        dcc.Checklist(
            id="observed-vars-checklist",
            options={
                "HFEV1": "HFEV1",
                "HO2Sat": "HO2Sat",
                "AR": "AR",
                "FEV1": "FEV1",
                "O2 saturation": "O2 saturation",
                "FEF25-75": "FEF25-75",
            },
            value=["FEV1", "O2 saturation"],
        ),
    ],
    style={"font-size": styles.font_size()},
)
