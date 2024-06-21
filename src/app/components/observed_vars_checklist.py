from dash import dcc, html

import src.app.assets.styles as styles

observed_vars_checklist_layout = html.Div(
    [
        html.Div("2. Select the observed variables:"),
        dcc.Checklist(
            id="observed-vars-checklist",
            options={
                "FEV1": "FEV1",
                "O2 saturation": "O2 saturation",
                "FEF25-75": "FEF25-75",
            },
            value=["FEV1", "O2 saturation"],
        ),
    ],
    style={"font-size": styles.font_size()},
)
