from typing import List
from dash import Input, Output

def show_slider_or_graph_for_observed_measures(app):
    @app.callback(
        Output("FEV1-dist", "style"),
        Output("FEV1-slider-container", "style"),
        Output("O2-saturation-dist", "style"),
        Output("O2-saturation-slider-container", "style"),
        Output("FEF25-75-dist", "style"),
        Output("FEF25-75-slider-container", "style"),
        Input("observed-vars-checklist", "value"),
    )
    def content(
        observed_vars_checklist: List[str],
    ):

        if "FEF25-75" in observed_vars_checklist:
            fef2575_style = {"display": "none"}, {"display": "block"}
        else:
            fef2575_style = {"display": "block"}, {"display": "none"}
        if "O2 saturation" in observed_vars_checklist:
            o2sat_style = {"display": "none"}, {"display": "block"}
        else:
            o2sat_style = {"display": "block"}, {"display": "none"}
        if "FEV1" in observed_vars_checklist:
            fev1_style = {"display": "none"}, {"display": "block"}
        else:
            fev1_style = {"display": "block"}, {"display": "none"}
        return *fev1_style, *o2sat_style, *fef2575_style