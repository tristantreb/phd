from typing import List

from dash import Input, Output


def show_slider_or_graph_for_observed_measures(app):
    @app.callback(
        Output("HFEV1-dist", "style"),
        Output("HFEV1-slider-container", "style"),
        Output("HO2Sat-dist", "style"),
        Output("HO2Sat-slider-container", "style"),
        Output("AR-dist", "style"),
        Output("AR-slider-container", "style"),
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

        def manage_obs_var(obs_var: str):
            if obs_var in observed_vars_checklist:
                return {"display": "none"}, {"display": "block"}
            else:
                return {"display": "block"}, {"display": "none"}

        hfev1_style = manage_obs_var("HFEV1")
        ho2sat_style = manage_obs_var("HO2Sat")
        ar_style = manage_obs_var("AR")
        fef2575_style = manage_obs_var("FEF25-75")
        o2sat_style = manage_obs_var("O2 saturation")
        fev1_style = manage_obs_var("FEV1")

        return (
            *hfev1_style,
            *ho2sat_style,
            *ar_style,
            *fev1_style,
            *o2sat_style,
            *fef2575_style,
        )
